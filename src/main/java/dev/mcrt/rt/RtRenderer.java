package dev.mcrt.rt;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vertex.DefaultVertexFormat;
import com.mojang.blaze3d.vertex.MeshData;
import com.mojang.renderpearl.backend.vulkan.VulkanCommandEncoder;
import com.mojang.renderpearl.backend.vulkan.VulkanConst;
import com.mojang.renderpearl.backend.vulkan.VulkanDevice;
import com.mojang.renderpearl.backend.vulkan.VulkanGpuTexture;
import dev.mcrt.McrtClient;
import dev.mcrt.mixin.FrontendGpuDeviceAccessor;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.client.renderer.state.level.LevelRenderState;
import net.minecraft.client.renderer.state.level.SkyRenderState;
import net.minecraft.client.renderer.texture.TextureAtlas;
import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.lwjgl.vulkan.VK;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkDevice;

/**
 * Owns the native path tracer. The world pass runs inside Minecraft's frame graph in place of
 * terrain and sky rasterization; it writes color and depth into the main target so entities,
 * particles and the hand still depth-test against the traced world.
 */
public final class RtRenderer {
	private static final long VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT = 0x00010000L;
	private static final RtRenderer INSTANCE = new RtRenderer();
	private static final int DEBUG_MODE = Integer.getInteger("mcrt.debug", 0);
	/** Light half the pixels per frame (alternating) and let the denoiser fill in; -Dmcrt.checkerboard=false to disable. */
	private static final boolean CHECKERBOARD = !"false".equals(System.getProperty("mcrt.checkerboard"));
	/** Far landscape past render distance (singleplayer); -Dmcrt.farTerrain=false turns it off. */
	private static final boolean FAR_TERRAIN = !"false".equals(System.getProperty("mcrt.farTerrain"));
	private final FarTerrain farTerrain = new FarTerrain();
	/** Internal render resolution relative to the window; the TAA pass upscales. -Dmcrt.renderScale=0.67 etc. */
	private static final float RENDER_SCALE = Math.clamp(Float.parseFloat(System.getProperty("mcrt.renderScale", "1.0")), 0.5f, 1.0f);

	private enum State { UNINITIALIZED, READY, DISABLED }

	private volatile State state = State.UNINITIALIZED;
	private NativeBridge bridge;
	private MemorySegment ctx;
	private MaterialRegistry materials = new MaterialRegistry();
	private VulkanDevice device;
	private final MemorySegment input = Arena.global().allocate(NativeBridge.FRAME_INPUT_SIZE, 16);
	private final MemorySegment output = Arena.global().allocate(NativeBridge.FRAME_OUTPUT_SIZE, 8);
	private final Matrix4f levelProjection = new Matrix4f();
	private boolean hasLevelProjection;
	private final Matrix4f viewProj = new Matrix4f();
	private final Matrix4f invViewProj = new Matrix4f();
	private final MemorySegment stats = Arena.global().allocate(NativeBridge.STATS_SIZE, 8);
	private final long startNanos = System.nanoTime();
	private int frameIndex;
	private long statsWindowStart = System.nanoTime();
	private int statsWindowFrames;

	private static volatile float[] keyLight;

	/** Direction towards the sun (or the moon at night) of the last frame, or null when neither lights the world. */
	public static float[] keyLightDirection() {
		return keyLight;
	}

	public static RtRenderer get() {
		return INSTANCE;
	}

	public boolean isActive() {
		return state == State.READY;
	}

	/** Called on the render thread before Minecraft (re)builds its section storage. */
	public void onGeometryInvalidated() {
		if (state == State.UNINITIALIZED) {
			initialize();
		}
		clearSections();
	}

	public void clearSections() {
		synchronized (SectionCapture.LOCK) {
			if (isActive()) {
				bridge.sectionsClear(ctx);
			}
			farTerrain.invalidate();
		}
	}

	/** Any thread: hands a sampled far landscape to the native renderer. */
	void submitFarTerrain(int originX, int originZ, int size, int spacing, int seaLevel, float[] heights, int[] colors) {
		synchronized (SectionCapture.LOCK) {
			if (!isActive()) {
				return;
			}
			try (Arena arena = Arena.ofConfined()) {
				MemorySegment[] segments = new MemorySegment[2];
				FarTerrain.copyTo(arena, heights, colors, segments);
				bridge.farTerrain(ctx, originX, originZ, size, spacing, seaLevel, segments[0], segments[1]);
			}
		}
	}

	/** The projection Minecraft rasterizes the level with (includes view bobbing). */
	public void setLevelProjection(Matrix4f projection) {
		levelProjection.set(projection);
		hasLevelProjection = true;
	}

	// Called with SectionCapture.LOCK held.
	MaterialRegistry materials() {
		return materials;
	}

	// Called with SectionCapture.LOCK held. scan is null for sections without geometry.
	void updateSection(int x, int y, int z, MeshData solid, MeshData cutout, MeshData translucent, SectionScanner.Result scan) {
		try (Arena arena = Arena.ofConfined()) {
			int[] emitters = scan == null ? new int[0] : scan.emitters();
			MemorySegment lights = emitters.length == 0 ? MemorySegment.NULL : arena.allocateFrom(JAVA_INT, emitters);
			MemorySegment blockMaterials = scan == null || scan.materials() == null
				? MemorySegment.NULL : arena.allocateFrom(JAVA_INT, scan.materials());
			MemorySegment occupancy = scan == null || scan.occupancy() == null
				? MemorySegment.NULL : arena.allocateFrom(java.lang.foreign.ValueLayout.JAVA_BYTE, scan.occupancy());
			bridge.sectionUpdate(ctx, x, y, z,
				vertices(solid), vertexCount(solid),
				vertices(cutout), vertexCount(cutout),
				vertices(translucent), vertexCount(translucent),
				lights, emitters.length, blockMaterials, occupancy);
		}
	}

	// Called with SectionCapture.LOCK held.
	void removeSection(int x, int y, int z) {
		bridge.sectionRemove(ctx, x, y, z);
	}

	private static boolean usable(MeshData mesh) {
		return mesh != null
			&& mesh.drawState().format() == DefaultVertexFormat.BLOCK
			&& mesh.vertexBuffer().remaining() == mesh.drawState().vertexCount() * NativeBridge.VERTEX_STRIDE;
	}

	private static MemorySegment vertices(MeshData mesh) {
		return usable(mesh) ? MemorySegment.ofBuffer(mesh.vertexBuffer()) : MemorySegment.NULL;
	}

	private static int vertexCount(MeshData mesh) {
		return usable(mesh) ? mesh.drawState().vertexCount() : 0;
	}

	public void renderWorld(LevelRenderState level, RenderTarget target) {
		if (!isActive()
			|| !(target.getColorTexture() instanceof VulkanGpuTexture color)
			|| !(target.getDepthTexture() instanceof VulkanGpuTexture depth)
			|| !(Minecraft.getInstance().getTextureManager().getTexture(TextureAtlas.LOCATION_BLOCKS).getTexture() instanceof VulkanGpuTexture atlas)) {
			return;
		}
		CameraRenderState camera = level.cameraRenderState;
		if (!camera.initialized) {
			return;
		}

		viewProj.set(hasLevelProjection ? levelProjection : camera.projectionMatrix).mul(camera.viewRotationMatrix);
		viewProj.invert(invViewProj);
		if (!invViewProj.isFinite()) {
			return; // projection not set up yet (first frames after joining a world)
		}

		int blockX = (int) Math.floor(camera.pos.x);
		int blockY = (int) Math.floor(camera.pos.y);
		int blockZ = (int) Math.floor(camera.pos.z);
		if (FAR_TERRAIN) {
			farTerrain.update(this, camera.pos.x, camera.pos.z);
		}
		int[] entityVertices = new int[1];
		float[] entityPositions = EntityCapture.take(blockX, blockY, blockZ, entityVertices);
		try (Arena arena = Arena.ofConfined()) {
			bridge.entities(ctx, entityVertices[0] == 0 ? MemorySegment.NULL : arena.allocateFrom(JAVA_FLOAT, entityPositions),
				entityVertices[0]);
		}

		input.set(JAVA_LONG, NativeBridge.OFF_COLOR_IMAGE, color.vkImage());
		input.set(JAVA_LONG, NativeBridge.OFF_DEPTH_IMAGE, depth.vkImage());
		input.set(JAVA_LONG, NativeBridge.OFF_ATLAS_IMAGE, atlas.vkImage());
		input.set(JAVA_INT, NativeBridge.OFF_COLOR_FORMAT, VulkanConst.toVk(color.getFormat()));
		input.set(JAVA_INT, NativeBridge.OFF_DEPTH_FORMAT, VulkanConst.toVk(depth.getFormat()));
		input.set(JAVA_INT, NativeBridge.OFF_ATLAS_FORMAT, VulkanConst.toVk(atlas.getFormat()));
		input.set(JAVA_INT, NativeBridge.OFF_ATLAS_MIP_LEVELS, atlas.getMipLevels());
		input.set(JAVA_INT, NativeBridge.OFF_WIDTH, color.getWidth(0));
		input.set(JAVA_INT, NativeBridge.OFF_HEIGHT, color.getHeight(0));
		input.set(JAVA_INT, NativeBridge.OFF_ATLAS_WIDTH, atlas.getWidth(0));
		input.set(JAVA_INT, NativeBridge.OFF_ATLAS_HEIGHT, atlas.getHeight(0));
		input.set(JAVA_INT, NativeBridge.OFF_FRAME_INDEX, frameIndex++);
		input.set(JAVA_INT, NativeBridge.OFF_DEBUG_MODE, DEBUG_MODE);
		input.set(JAVA_INT, NativeBridge.OFF_CAMERA_BLOCK_POS, blockX);
		input.set(JAVA_INT, NativeBridge.OFF_CAMERA_BLOCK_POS + 4, blockY);
		input.set(JAVA_INT, NativeBridge.OFF_CAMERA_BLOCK_POS + 8, blockZ);
		input.set(JAVA_INT, NativeBridge.OFF_FLAGS, frameFlags(level));
		putVec4(NativeBridge.OFF_CAMERA_OFFSET,
			(float) (camera.pos.x - blockX), (float) (camera.pos.y - blockY), (float) (camera.pos.z - blockZ), 0f);
		putMatrix(NativeBridge.OFF_VIEW_PROJ, viewProj);
		putMatrix(NativeBridge.OFF_INV_VIEW_PROJ, invViewProj);
		putSky(level.skyRenderState);
		if (level.skyRenderState.skybox != net.minecraft.world.level.dimension.DimensionType.Skybox.OVERWORLD
			&& camera.fogData != null && camera.fogData.color != null) {
			// Dimensions without a sky: their fog color stands in for the light from "outside".
			putVec4(NativeBridge.OFF_SKY_COLOR, camera.fogData.color.x, camera.fogData.color.y, camera.fogData.color.z, 1f);
		}
		input.set(JAVA_FLOAT, NativeBridge.OFF_TIME, (System.nanoTime() - startNanos) * 1e-9f);
		// Angle one pixel subtends at the screen center; drives texture LOD selection.
		Matrix4f projection = hasLevelProjection ? levelProjection : camera.projectionMatrix;
		input.set(JAVA_FLOAT, NativeBridge.OFF_PIXEL_SPREAD, 2f / (color.getHeight(0) * Math.abs(projection.m11())));
		input.set(JAVA_FLOAT, NativeBridge.OFF_CLOUD_HEIGHT, level.cloudHeight);
		input.set(JAVA_FLOAT, NativeBridge.OFF_RENDER_SCALE, RENDER_SCALE);
		input.set(JAVA_FLOAT, NativeBridge.OFF_RENDER_DISTANCE, Minecraft.getInstance().options.getEffectiveRenderDistance() * 16.0f);

		int result = bridge.renderFrame(ctx, input, output);
		if (result < 0) {
			disable("Native frame failed: " + bridge.lastError());
			return;
		}
		if (result == 0) {
			return;
		}

		logStatsPeriodically();
		VulkanCommandEncoder encoder = device.createCommandEncoder();
		encoder.execute(new VkCommandBuffer(output.get(JAVA_LONG, NativeBridge.OFF_OUT_COMMAND_BUFFER), device.vkDevice()));
		encoder.signalSemaphore(
			output.get(JAVA_LONG, NativeBridge.OFF_OUT_SEMAPHORE),
			output.get(JAVA_LONG, NativeBridge.OFF_OUT_SIGNAL_VALUE),
			VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
		);
	}

	/** Vanilla-comparison runs (-Dmcrt.disable) still log frame rate. */
	public static void logVanillaFps() {
		INSTANCE.statsWindowFrames++;
		long now = System.nanoTime();
		if (now - INSTANCE.statsWindowStart >= 5_000_000_000L) {
			McrtClient.LOGGER.info("[vanilla] fps={}", String.format("%.1f",
				INSTANCE.statsWindowFrames / ((now - INSTANCE.statsWindowStart) * 1e-9)));
			INSTANCE.statsWindowStart = now;
			INSTANCE.statsWindowFrames = 0;
		}
	}

	private void logStatsPeriodically() {
		statsWindowFrames++;
		long now = System.nanoTime();
		if (now - statsWindowStart < 5_000_000_000L) {
			return;
		}
		bridge.getStats(ctx, stats);
		McrtClient.LOGGER.info("[stats] fps={} gpuPass={}ms nativeCpu={}ms lightRebuild={}ms sections={} pending={} instances={} lights={}",
			String.format("%.1f", statsWindowFrames / ((now - statsWindowStart) * 1e-9)),
			String.format("%.2f", stats.get(JAVA_FLOAT, 16)),
			String.format("%.2f", stats.get(JAVA_FLOAT, 24)), String.format("%.2f", stats.get(JAVA_FLOAT, 28)),
			stats.get(JAVA_INT, 0), stats.get(JAVA_INT, 4), stats.get(JAVA_INT, 8), stats.get(JAVA_INT, 20));
		McrtClient.LOGGER.info("[sky] sun=({}, {}, {}, i={}) moon=({}, {}, {}, i={}) skyColor=({}, {}, {}) rain={}",
			input.get(JAVA_FLOAT, NativeBridge.OFF_SUN_DIR), input.get(JAVA_FLOAT, NativeBridge.OFF_SUN_DIR + 4),
			input.get(JAVA_FLOAT, NativeBridge.OFF_SUN_DIR + 8), input.get(JAVA_FLOAT, NativeBridge.OFF_SUN_DIR + 12),
			input.get(JAVA_FLOAT, NativeBridge.OFF_MOON_DIR), input.get(JAVA_FLOAT, NativeBridge.OFF_MOON_DIR + 4),
			input.get(JAVA_FLOAT, NativeBridge.OFF_MOON_DIR + 8), input.get(JAVA_FLOAT, NativeBridge.OFF_MOON_DIR + 12),
			input.get(JAVA_FLOAT, NativeBridge.OFF_SKY_COLOR), input.get(JAVA_FLOAT, NativeBridge.OFF_SKY_COLOR + 4),
			input.get(JAVA_FLOAT, NativeBridge.OFF_SKY_COLOR + 8), input.get(JAVA_FLOAT, NativeBridge.OFF_RAIN));
		statsWindowStart = now;
		statsWindowFrames = 0;
	}

	// Bits 0-1 sky type (0 overworld, 1 none/Nether, 2 End); bit 2 camera in water; bit 3 camera in lava;
	// bit 4 checkerboard lighting.
	private static int frameFlags(LevelRenderState level) {
		int flags = CHECKERBOARD ? 16 : 0;
		flags |= switch (level.skyRenderState.skybox) {
			case OVERWORLD -> 0;
			case END -> 2;
			default -> 1;
		};
		net.minecraft.world.level.material.FogType fog = level.cameraRenderState.fogType;
		if (fog == net.minecraft.world.level.material.FogType.WATER) {
			flags |= 4;
		} else if (fog == net.minecraft.world.level.material.FogType.LAVA) {
			flags |= 8;
		}
		return flags;
	}

	private void putSky(SkyRenderState sky) {
		// Same rotation SkyRenderer applies to the sun quad at (0, 100, 0): Ry(-90deg) * Rx(angle).
		Vector3f sun = new Vector3f(0f, 1f, 0f).rotateX(sky.sunAngle).rotateY((float) (-Math.PI / 2));
		Vector3f moon = new Vector3f(0f, 1f, 0f).rotateX(sky.moonAngle).rotateY((float) (-Math.PI / 2));
		float sunIntensity = smoothstep(-0.05f, 0.12f, sun.y);
		float moonIntensity = smoothstep(-0.05f, 0.12f, moon.y) * (1f - sunIntensity);
		putVec4(NativeBridge.OFF_SUN_DIR, sun.x, sun.y, sun.z, sunIntensity);
		keyLight = sunIntensity > 0.05f ? new float[] {sun.x, sun.y, sun.z}
			: moonIntensity > 0.05f ? new float[] {moon.x, moon.y, moon.z} : null;
		putVec4(NativeBridge.OFF_MOON_DIR, moon.x, moon.y, moon.z, moonIntensity);
		if (sky.skyColor != null) {
			putVec4(NativeBridge.OFF_SKY_COLOR, sky.skyColor.x(), sky.skyColor.y(), sky.skyColor.z(), 1f);
		} else {
			putVec4(NativeBridge.OFF_SKY_COLOR, 0.47f, 0.65f, 1f, 1f);
		}
		input.set(JAVA_FLOAT, NativeBridge.OFF_RAIN, Math.clamp(1f - sky.rainBrightness, 0f, 1f));
	}

	private static float smoothstep(float edge0, float edge1, float x) {
		float t = Math.clamp((x - edge0) / (edge1 - edge0), 0f, 1f);
		return t * t * (3f - 2f * t);
	}

	private void putVec4(long offset, float x, float y, float z, float w) {
		input.set(JAVA_FLOAT, offset, x);
		input.set(JAVA_FLOAT, offset + 4, y);
		input.set(JAVA_FLOAT, offset + 8, z);
		input.set(JAVA_FLOAT, offset + 12, w);
	}

	// Column-major, matching JOML's storage order.
	private void putMatrix(long offset, Matrix4f m) {
		for (int col = 0; col < 4; col++) {
			for (int row = 0; row < 4; row++) {
				input.set(JAVA_FLOAT, offset + (col * 4L + row) * 4L, m.get(col, row));
			}
		}
	}

	private void initialize() {
		if (Boolean.getBoolean("mcrt.disable")) {
			disable("disabled by -Dmcrt.disable (vanilla rendering for comparison)");
			return;
		}
		if (!(RenderSystem.getDevice() instanceof FrontendGpuDeviceAccessor frontend)
			|| !(frontend.mcrt$getBackend() instanceof VulkanDevice vulkanDevice)) {
			disable("Minecraft is not running on the Vulkan backend. Set Graphics API to Vulkan in Video Settings.");
			return;
		}
		try {
			bridge = NativeBridge.load();
			VkDevice vkDevice = vulkanDevice.vkDevice();
			long gipa = VK.getFunctionProvider().getFunctionAddress("vkGetInstanceProcAddr");
			MemorySegment created = bridge.create(
				gipa,
				vkDevice.getPhysicalDevice().getInstance().address(),
				vkDevice.getPhysicalDevice().address(),
				vkDevice.address(),
				vulkanDevice.graphicsQueue().queueFamilyIndex()
			);
			if (created.address() == 0) {
				disable("Native renderer creation failed: " + bridge.lastError());
				return;
			}
			ctx = created;
			device = vulkanDevice;
			materials = MaterialRegistry.load();
			materials.upload(bridge, ctx);
			state = State.READY;
			McrtClient.LOGGER.info("MCRT native path tracer initialized");
		} catch (Exception | LinkageError e) {
			McrtClient.LOGGER.error("MCRT failed to load native renderer", e);
			state = State.DISABLED;
		}
	}

	/** Must run before Minecraft destroys the Vulkan device. */
	public void shutdown() {
		synchronized (SectionCapture.LOCK) {
			if (state == State.READY) {
				state = State.DISABLED;
				bridge.destroy(ctx);
				ctx = null;
				device = null;
			}
			state = State.DISABLED;
		}
	}

	private void disable(String reason) {
		McrtClient.LOGGER.error("MCRT disabled: {}", reason);
		state = State.DISABLED;
	}
}
