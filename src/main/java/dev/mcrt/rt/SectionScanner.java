package dev.mcrt.rt;

import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import net.minecraft.client.renderer.chunk.RenderSectionRegion;
import net.minecraft.core.BlockPos;
import net.minecraft.core.SectionPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;

/**
 * Per-block data for a section being compiled: light emitters and PBR material assignments.
 * Local block index = (y << 8) | (z << 4) | x.
 */
final class SectionScanner {
	private static final Map<Block, Integer> LIGHT_COLORS = new ConcurrentHashMap<>();

	/**
	 * @param emitters  each: bits 0-11 local index, 12-15 emission level, 16-31 RGB565 color
	 * @param materials 4096 packed face materials (see MaterialRegistry), or null if none in the section
	 */
	record Result(int[] emitters, int[] materials) {
	}

	private SectionScanner() {
	}

	static Result scan(RenderSectionRegion region, SectionPos section, MaterialRegistry registry) {
		int[] emitters = null;
		int emitterCount = 0;
		int[] materials = null;
		BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
		int baseX = section.minBlockX(), baseY = section.minBlockY(), baseZ = section.minBlockZ();
		for (int y = 0; y < 16; y++) {
			for (int z = 0; z < 16; z++) {
				for (int x = 0; x < 16; x++) {
					BlockState state = region.getBlockState(pos.set(baseX + x, baseY + y, baseZ + z));
					int local = (y << 8) | (z << 4) | x;
					int faces = registry.faces(state.getBlock());
					if (faces != 0) {
						if (materials == null) {
							materials = new int[4096];
						}
						materials[local] = faces;
					}
					int emission = state.getLightEmission();
					if (emission > 0) {
						if (emitters == null) {
							emitters = new int[64];
						} else if (emitterCount == emitters.length) {
							emitters = Arrays.copyOf(emitters, emitterCount * 2);
						}
						int color = LIGHT_COLORS.computeIfAbsent(state.getBlock(), SectionScanner::lightColor);
						emitters[emitterCount++] = local | (Math.min(emission, 15) << 12) | (color << 16);
					}
				}
			}
		}
		return new Result(emitters == null ? new int[0] : Arrays.copyOf(emitters, emitterCount), materials);
	}

	/** Light color by block id; Minecraft has no light color, so this is a hand-made palette. */
	private static int lightColor(Block block) {
		String id = BuiltInRegistries.BLOCK.getKey(block).getPath();
		if (id.contains("soul")) return rgb565(110, 210, 255);
		if (id.contains("redstone")) return rgb565(255, 60, 40);
		if (id.contains("lava") || id.contains("magma")) return rgb565(255, 110, 30);
		if (id.contains("copper_torch") || id.contains("copper_lantern")) return rgb565(140, 255, 140);
		if (id.contains("sea_lantern") || id.contains("beacon") || id.contains("conduit")) return rgb565(190, 230, 255);
		if (id.contains("end_rod")) return rgb565(240, 230, 255);
		if (id.contains("crying_obsidian") || id.contains("amethyst") || id.contains("respawn_anchor") || id.contains("portal")) return rgb565(170, 80, 255);
		if (id.contains("sculk")) return rgb565(40, 200, 220);
		if (id.contains("pearlescent_froglight")) return rgb565(255, 200, 240);
		if (id.contains("verdant_froglight")) return rgb565(200, 255, 180);
		if (id.contains("glowstone") || id.contains("shroomlight") || id.contains("froglight")) return rgb565(255, 200, 130);
		return rgb565(255, 170, 90); // torches, lanterns, fire, campfires, glow lichen...
	}

	private static int rgb565(int r, int g, int b) {
		return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
	}
}
