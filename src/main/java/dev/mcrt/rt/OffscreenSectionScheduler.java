package dev.mcrt.rt;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import net.minecraft.client.SectionUpdateTracker;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.ViewArea;
import net.minecraft.client.renderer.chunk.CompiledSectionMesh;
import net.minecraft.client.renderer.chunk.RenderRegionCache;
import net.minecraft.client.renderer.chunk.SectionRenderDispatcher;
import net.minecraft.client.renderer.state.level.SectionUpdateRenderState;
import net.minecraft.core.BlockPos;
import net.minecraft.core.SectionPos;

/**
 * Walks the whole view area a slice at a time (nearest columns first) and schedules dirty
 * sections Minecraft skipped because they were not visible. Budgeted per frame, since creating
 * a compile region copies chunk data on the render thread.
 */
public final class OffscreenSectionScheduler {
	private static final int COLUMNS_PER_FRAME = 96;
	private static final int MAX_SCHEDULED_PER_FRAME = 16;

	private int viewDistance = -1;
	private List<int[]> columnOffsets = List.of();
	private int cursor;

	public void schedule(ViewArea viewArea, ClientLevel level, SectionUpdateTracker tracker, BlockPos cameraPos,
			List<SectionUpdateRenderState> output) {
		if (viewArea == null) {
			return;
		}
		if (viewArea.getViewDistance() != viewDistance) {
			rebuildOffsets(viewArea.getViewDistance());
		}

		int cameraX = SectionPos.blockToSectionCoord(cameraPos.getX());
		int cameraZ = SectionPos.blockToSectionCoord(cameraPos.getZ());
		RenderRegionCache cache = null;
		int scheduled = 0;
		BlockPos.MutableBlockPos origin = new BlockPos.MutableBlockPos();

		for (int i = 0; i < COLUMNS_PER_FRAME && scheduled < MAX_SCHEDULED_PER_FRAME; i++) {
			int[] offset = columnOffsets.get(cursor);
			cursor = (cursor + 1) % columnOffsets.size();
			int sectionX = cameraX + offset[0];
			int sectionZ = cameraZ + offset[1];
			for (int sectionY = viewArea.minSectionY(); sectionY <= viewArea.maxSectionY(); sectionY++) {
				long node = SectionPos.asLong(sectionX, sectionY, sectionZ);
				SectionUpdateTracker.SectionDirtyState dirty = tracker.getDirtyState(node);
				if (dirty == null || !dirty.isDirty()) {
					continue;
				}
				origin.set(SectionPos.sectionToBlockCoord(sectionX), SectionPos.sectionToBlockCoord(sectionY), SectionPos.sectionToBlockCoord(sectionZ));
				SectionRenderDispatcher.RenderSection section = viewArea.getRenderSectionAt(origin);
				if (section == null || section.getSectionNode() != node) {
					continue;
				}
				// Same readiness rule Minecraft applies to visible sections.
				if (section.sectionMesh.get() == CompiledSectionMesh.UNCOMPILED && !tracker.hasAllNeighbors(level, node)) {
					continue;
				}
				if (cache == null) {
					cache = new RenderRegionCache();
				}
				output.add(new SectionUpdateRenderState(node, dirty.isDirtyFromPlayer(), cache.createRegion(level, node)));
				dirty.setNotDirty();
				scheduled++;
			}
		}
	}

	private void rebuildOffsets(int distance) {
		List<int[]> offsets = new ArrayList<>();
		for (int dx = -distance; dx <= distance; dx++) {
			for (int dz = -distance; dz <= distance; dz++) {
				offsets.add(new int[] {dx, dz});
			}
		}
		offsets.sort(Comparator.comparingInt(o -> o[0] * o[0] + o[1] * o[1]));
		columnOffsets = offsets;
		viewDistance = distance;
		cursor = 0;
	}
}
