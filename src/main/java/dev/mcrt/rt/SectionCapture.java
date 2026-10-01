package dev.mcrt.rt;

import com.mojang.blaze3d.vertex.MeshData;
import java.util.Map;
import net.minecraft.client.renderer.chunk.ChunkSectionLayer;
import net.minecraft.client.renderer.chunk.RenderSectionRegion;
import net.minecraft.client.renderer.chunk.SectionCompiler;
import net.minecraft.core.SectionPos;

/**
 * Forwards Minecraft's compiled section meshes to the native renderer.
 *
 * <p>Sections compile on worker threads while the render thread may reset (unload or move) them.
 * A compile that finishes after its section was reset must not resurrect stale geometry. The
 * expensive native preparation runs without locks (the render thread reassigns sections constantly
 * while the player moves and must never wait for it); only the final, cheap commit happens under
 * the lock, and only if the section is still assigned exactly as it was when the work began.
 */
public final class SectionCapture {
	static final Object LOCK = new Object();
	/** Live sections and their assignment epoch (bumped whenever a section is (re)assigned). */
	private static final it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap LIVE = new it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap();
	private static int nextEpoch = 1;

	private SectionCapture() {
	}

	public static void onAssigned(long sectionNode) {
		synchronized (LOCK) {
			LIVE.put(sectionNode, nextEpoch++);
		}
	}

	public static void onReset(long sectionNode) {
		synchronized (LOCK) {
			if (LIVE.containsKey(sectionNode) && LIVE.remove(sectionNode) != 0 && RtRenderer.get().isActive()) {
				RtRenderer.get().removeSection(SectionPos.x(sectionNode), SectionPos.y(sectionNode), SectionPos.z(sectionNode));
			}
		}
	}

	public static void onCompiled(SectionPos pos, RenderSectionRegion region, SectionCompiler.Results results) {
		RtRenderer renderer = RtRenderer.get();
		if (!renderer.isActive()) {
			return;
		}
		long node = pos.asLong();
		int epoch;
		synchronized (LOCK) {
			epoch = LIVE.get(node);
		}
		if (epoch == 0) {
			return;
		}
		Map<ChunkSectionLayer, MeshData> layers = results.renderedLayers;
		SectionScanner.Result scan = layers.isEmpty() ? null : SectionScanner.scan(region, pos, renderer.materials());
		long prepared = renderer.prepareSection(pos.x(), pos.y(), pos.z(),
			layers.get(ChunkSectionLayer.SOLID),
			layers.get(ChunkSectionLayer.CUTOUT),
			layers.get(ChunkSectionLayer.TRANSLUCENT),
			scan);
		synchronized (LOCK) {
			renderer.commitSection(prepared, LIVE.get(node) == epoch);
		}
	}
}
