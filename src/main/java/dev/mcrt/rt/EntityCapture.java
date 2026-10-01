package dev.mcrt.rt;

import com.mojang.blaze3d.vertex.MeshData;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import net.minecraft.world.phys.Vec3;

/**
 * Collects the entity quads Minecraft prepares for the level (between {@link #begin} and
 * {@link #end}, i.e. inside LevelRenderer's feature preparation, so GUI models stay out) and
 * hands them to the ray tracer for shadows, reflections and bounce light.
 *
 * <p>Entity vertices are camera-relative; they are stored relative to the camera's block.
 */
public final class EntityCapture {
	private static final int MAX_VERTICES = 4 * 100_000;
	private static boolean active;
	private static double offsetX, offsetY, offsetZ; // camera position minus its block
	private static int blockX, blockY, blockZ;
	private static float[] positions = new float[3 * 4096];
	private static int vertexCount;

	private EntityCapture() {
	}

	public static void begin(Vec3 camera) {
		blockX = (int) Math.floor(camera.x);
		blockY = (int) Math.floor(camera.y);
		blockZ = (int) Math.floor(camera.z);
		offsetX = camera.x - blockX;
		offsetY = camera.y - blockY;
		offsetZ = camera.z - blockZ;
		vertexCount = 0;
		active = true;
	}

	public static void end() {
		active = false;
	}

	public static void capture(MeshData mesh, int vertexSize) {
		if (!active) {
			return;
		}
		int count = mesh.drawState().vertexCount() & ~3;
		if (count == 0 || vertexCount + count > MAX_VERTICES) {
			return;
		}
		ByteBuffer buffer = mesh.vertexBuffer().duplicate().order(ByteOrder.nativeOrder());
		if (positions.length < (vertexCount + count) * 3) {
			positions = java.util.Arrays.copyOf(positions, Math.max(positions.length * 2, (vertexCount + count) * 3));
		}
		int base = buffer.position();
		for (int v = 0; v < count; v++) {
			int at = base + v * vertexSize;
			int out = (vertexCount + v) * 3;
			positions[out] = (float) (buffer.getFloat(at) + offsetX);
			positions[out + 1] = (float) (buffer.getFloat(at + 4) + offsetY);
			positions[out + 2] = (float) (buffer.getFloat(at + 8) + offsetZ);
		}
		vertexCount += count;
	}

	/** Render thread: the quads of the frame being rendered, relative to the given camera block. */
	static float[] take(int cameraBlockX, int cameraBlockY, int cameraBlockZ, int[] countOut) {
		float dx = blockX - cameraBlockX, dy = blockY - cameraBlockY, dz = blockZ - cameraBlockZ;
		float[] out = new float[vertexCount * 3];
		for (int i = 0; i < vertexCount; i++) {
			out[i * 3] = positions[i * 3] + dx;
			out[i * 3 + 1] = positions[i * 3 + 1] + dy;
			out[i * 3 + 2] = positions[i * 3 + 2] + dz;
		}
		countOut[0] = vertexCount;
		return out;
	}
}
