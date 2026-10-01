package dev.mcrt.rt;

import com.mojang.blaze3d.vertex.MeshData;
import it.unimi.dsi.fastutil.longs.LongOpenHashSet;
import java.util.Map;
import net.minecraft.client.renderer.chunk.ChunkSectionLayer;
import net.minecraft.client.renderer.chunk.RenderSectionRegion;
import net.minecraft.client.renderer.chunk.SectionCompiler;
import net.minecraft.core.SectionPos;

/**
 * Forwards Minecraft's compiled section meshes to the native renderer.
 *
 * <p>Sections compile on worker threads while the render thread may reset (unload or move) them.
 * A compile that finishes after its section was reset must not resurrect stale geometry, so the
 * live-section check and the native enqueue happen under one lock, as does every removal.
 */
public final class SectionCapture {
	static final Object LOCK = new Object();
	private static final LongOpenHashSet LIVE = new LongOpenHashSet();

	private SectionCapture() {
	}

	public static void onAssigned(long sectionNode) {
		synchronized (LOCK) {
			LIVE.add(sectionNode);
		}
	}

	public static void onReset(long sectionNode) {
		synchronized (LOCK) {
			if (LIVE.remove(sectionNode) && RtRenderer.get().isActive()) {
				RtRenderer.get().removeSection(SectionPos.x(sectionNode), SectionPos.y(sectionNode), SectionPos.z(sectionNode));
			}
		}
	}

	public static void onCompiled(SectionPos pos, RenderSectionRegion region, SectionCompiler.Results results) {
		RtRenderer renderer = RtRenderer.get();
		if (!renderer.isActive()) {
			return;
		}
		Map<ChunkSectionLayer, MeshData> layers = results.renderedLayers;
		SectionScanner.Result scan = layers.isEmpty() ? null : SectionScanner.scan(region, pos, renderer.materials());
		synchronized (LOCK) {
			if (!LIVE.contains(pos.asLong()) || !renderer.isActive()) {
				return;
			}
			renderer.updateSection(pos.x(), pos.y(), pos.z(),
				layers.get(ChunkSectionLayer.SOLID),
				layers.get(ChunkSectionLayer.CUTOUT),
				layers.get(ChunkSectionLayer.TRANSLUCENT),
				scan);
		}
	}
}
