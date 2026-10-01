package dev.mcrt.rt;

import dev.mcrt.McrtClient;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import net.minecraft.client.Minecraft;
import net.minecraft.client.server.IntegratedServer;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Holder;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.BiomeResolver;
import net.minecraft.world.level.chunk.ChunkGenerator;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.level.levelgen.RandomState;

/**
 * The landscape past render distance, so the view reaches the horizon instead of ending in haze.
 *
 * <p>In singleplayer the world generator is at hand: a background thread samples its terrain
 * height and biomes on a coarse grid around the player (no caves, structures or player edits,
 * which are invisible from that far anyway) and hands the heightfield to the native renderer.
 * The shader hides it wherever real chunks are loaded. Multiplayer has no generator: the loaded
 * world then fades into the haze as before.
 */
final class FarTerrain {
	static final int SIZE = 257;       // samples per side
	static final int SPACING = 16;     // blocks between samples: covers +-2048 blocks
	static final int RECENTER = 384;   // regenerate once the player is this far from the center
	static final int HELPERS = 3;      // extra sampling threads

	private final ExecutorService worker = Executors.newSingleThreadExecutor(r -> {
		Thread thread = new Thread(r, "MCRT far terrain");
		thread.setDaemon(true);
		thread.setPriority(Thread.MIN_PRIORITY);
		return thread;
	});
	private Future<?> task;
	private boolean valid;
	private int centerX, centerZ;
	private IntegratedServer server;

	/** The renderer dropped all sections (level change, reload): the far landscape went with them. */
	void invalidate() {
		valid = false;
	}

	/** Render thread, every frame. */
	void update(RtRenderer renderer, double cameraX, double cameraZ) {
		Minecraft minecraft = Minecraft.getInstance();
		IntegratedServer current = minecraft.getSingleplayerServer();
		if (current == null || minecraft.level == null || minecraft.level.dimension() != Level.OVERWORLD) {
			valid = valid && current == server && minecraft.level != null && minecraft.level.dimension() == Level.OVERWORLD;
			return;
		}
		if (current != server) {
			server = current;
			valid = false;
		}
		if (task != null && !task.isDone()) {
			return;
		}
		if (valid && Math.abs(cameraX - centerX) < RECENTER && Math.abs(cameraZ - centerZ) < RECENTER) {
			return;
		}
		int x = Math.floorDiv((int) Math.floor(cameraX), 64) * 64;
		int z = Math.floorDiv((int) Math.floor(cameraZ), 64) * 64;
		centerX = x;
		centerZ = z;
		valid = true;
		ServerLevel level = current.overworld();
		task = worker.submit(() -> {
			try {
				generate(renderer, level, x, z);
			} catch (RuntimeException e) {
				McrtClient.LOGGER.warn("MCRT: far terrain generation failed", e);
			}
		});
	}

	private static void generate(RtRenderer renderer, ServerLevel level, int originX, int originZ) {
		long start = System.nanoTime();
		ChunkGenerator generator = level.getChunkSource().getGenerator();
		RandomState random = level.getChunkSource().randomState();
		BiomeResolver biomes = generator.getBiomeSource().createUncachedResolver(random);
		int seaLevel = generator.getSeaLevel();
		float[] heights = new float[SIZE * SIZE];
		int[] colors = new int[SIZE * SIZE];
		int half = SIZE / 2;
		// Rows in parallel on a few low-priority threads (the generator is thread-safe: chunk
		// generation runs on many workers).
		java.util.concurrent.atomic.AtomicInteger nextRow = new java.util.concurrent.atomic.AtomicInteger();
		Runnable rows = () -> {
			BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
			for (int j = nextRow.getAndIncrement(); j < SIZE; j = nextRow.getAndIncrement()) {
				for (int i = 0; i < SIZE; i++) {
					int x = originX + (i - half) * SPACING;
					int z = originZ + (j - half) * SPACING;
					int y = generator.getBaseHeight(x, z, Heightmap.Types.OCEAN_FLOOR_WG, level, random);
					Holder<Biome> biome = biomes.getNoiseBiome(x >> 2, Math.max(y, seaLevel) >> 2, z >> 2);
					heights[j * SIZE + i] = y;
					colors[j * SIZE + i] = surfaceColor(biome, pos.set(x, y, z), y, seaLevel);
				}
			}
		};
		Thread[] helpers = new Thread[HELPERS];
		for (int t = 0; t < HELPERS; t++) {
			helpers[t] = new Thread(rows, "MCRT far terrain " + t);
			helpers[t].setDaemon(true);
			helpers[t].setPriority(Thread.MIN_PRIORITY);
			helpers[t].start();
		}
		rows.run();
		for (Thread helper : helpers) {
			try {
				helper.join();
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
				return;
			}
		}
		long sampled = System.nanoTime();
		renderer.submitFarTerrain(originX, originZ, SIZE, SPACING, seaLevel, heights, colors);
		McrtClient.LOGGER.info("MCRT: far terrain around ({}, {}) sampled in {} ms", originX, originZ,
			(sampled - start) / 1_000_000);
	}

	/** RGBA8 (r in the low byte) sRGB surface color as seen from afar; alpha 1 marks water. */
	private static int surfaceColor(Holder<Biome> holder, BlockPos pos, int height, int seaLevel) {
		Biome biome = holder.value();
		// getBaseHeight is the first free block above the surface: below sea level that is water.
		if (height < seaLevel) {
			float depth = Math.min((seaLevel - height) / 24.0f, 1.0f);
			return rgba(lerp(55, 14, depth), lerp(115, 40, depth), lerp(120, 66, depth), 1);
		}
		String path = holder.unwrapKey().map(key -> key.identifier().getPath()).orElse("");
		if (biome.coldEnoughToSnow(pos, seaLevel) || path.contains("frozen") || path.contains("snowy")) {
			return rgba(232, 236, 242, 0);
		}
		if (path.contains("desert") || path.contains("beach")) {
			return rgba(214, 198, 152, 0);
		}
		if (path.contains("badlands")) {
			return rgba(176, 98, 58, 0);
		}
		if (path.contains("peaks") || path.contains("stony") || height > 160) {
			return rgba(124, 120, 114, 0);
		}
		int grass = biome.getGrassColor(pos.getX(), pos.getZ());
		boolean wooded = path.contains("forest") || path.contains("taiga") || path.contains("jungle")
			|| path.contains("grove") || path.contains("swamp") || path.contains("woodland");
		int tint = wooded ? biome.getFoliageColor() : grass;
		float shade = wooded ? 0.5f : 0.78f; // canopies and grass are darker than their flat color
		return rgba((int) (((tint >> 16) & 0xFF) * shade), (int) (((tint >> 8) & 0xFF) * shade), (int) ((tint & 0xFF) * shade), 0);
	}

	private static int lerp(int a, int b, float t) {
		return Math.round(a + (b - a) * t);
	}

	private static int rgba(int r, int g, int b, int a) {
		return (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16) | ((a & 0xFF) << 24);
	}

	static void copyTo(Arena arena, float[] heights, int[] colors, MemorySegment[] out) {
		out[0] = arena.allocateFrom(ValueLayout.JAVA_FLOAT, heights);
		out[1] = arena.allocateFrom(ValueLayout.JAVA_INT, colors);
	}
}
