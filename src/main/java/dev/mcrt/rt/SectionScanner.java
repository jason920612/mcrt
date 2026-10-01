package dev.mcrt.rt;

import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import net.minecraft.client.renderer.chunk.RenderSectionRegion;
import net.minecraft.core.BlockPos;
import net.minecraft.core.SectionPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.SnowLayerBlock;
import net.minecraft.world.level.block.state.BlockState;

/**
 * Per-block data for a section being compiled: light emitters, PBR material assignments and the
 * terrain classification the native side smooths. Local block index = (y << 8) | (z << 4) | x.
 */
final class SectionScanner {
	static final byte OPEN = 0;    // air, plants, fluids, partial blocks: smooth terrain faces these
	static final byte SMOOTH = 1;  // natural block rendered as smoothed terrain
	static final byte SOLID = 2;   // other opaque full block: counts as inside, pins nearby corners
	static final byte COVER = 3;   // thin snow on smooth terrain: dropped, the surface below turns to snow
	static final byte PIN = 4;     // see-through full block (ice, glass, leaves): faced like open, pins corners
	// Snow layers up to this many fold into the surface below; deeper snow becomes a smooth snow block.
	private static final int MAX_COVER_LAYERS = 3;
	// Material id reserved for water surfaces (all faces); shaders render it as physical water.
	static final int WATER_MATERIAL = 255;
	private static final int WATER_FACES = WATER_MATERIAL | (WATER_MATERIAL << 8) | (WATER_MATERIAL << 16);
	static final int PAD = 2;      // occupancy covers the section plus PAD blocks on every side
	static final int BORDER = 16 + 2 * PAD;

	private static final Map<Block, Integer> LIGHT_COLORS = new ConcurrentHashMap<>();

	/**
	 * @param emitters  each: bits 0-11 local index, 12-15 emission level, 16-31 RGB565 color
	 * @param materials 4096 packed face materials (see MaterialRegistry), or null if none in the section
	 * @param occupancy 20^3 classification ((y+2)*400 + (z+2)*20 + (x+2)), or null without smooth blocks
	 */
	record Result(int[] emitters, int[] materials, byte[] occupancy) {
	}

	private SectionScanner() {
	}

	static Result scan(RenderSectionRegion region, SectionPos section, MaterialRegistry registry) {
		int[] emitters = null;
		int emitterCount = 0;
		int[] materials = null;
		boolean anySmooth = false;
		BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
		int baseX = section.minBlockX(), baseY = section.minBlockY(), baseZ = section.minBlockZ();
		for (int y = 0; y < 16; y++) {
			for (int z = 0; z < 16; z++) {
				for (int x = 0; x < 16; x++) {
					BlockState state = region.getBlockState(pos.set(baseX + x, baseY + y, baseZ + z));
					int local = (y << 8) | (z << 4) | x;
					int faces = registry.faces(state.getBlock());
					net.minecraft.world.level.material.Fluid fluid = state.getFluidState().getType();
					if (fluid.isSame(net.minecraft.world.level.material.Fluids.WATER)) {
						faces = WATER_FACES;
					}
					if (faces != 0) {
						if (materials == null) {
							materials = new int[4096];
						}
						materials[local] = faces;
						anySmooth |= faces != WATER_FACES && (registry.isSmooth(state.getBlock()) || isDeepSnow(state));
					}
					int emission = state.getLightEmission();
					if (emission > 0 && isExposed(region, pos, state)) {
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

		byte[] occupancy = null;
		if (anySmooth) {
			occupancy = new byte[BORDER * BORDER * BORDER];
			for (int y = -PAD; y < 16 + PAD; y++) {
				for (int z = -PAD; z < 16 + PAD; z++) {
					for (int x = -PAD; x < 16 + PAD; x++) {
						occupancy[(y + PAD) * BORDER * BORDER + (z + PAD) * BORDER + (x + PAD)] =
							classify(region, pos, baseX + x, baseY + y, baseZ + z, registry);
					}
				}
			}
			// Smooth blocks under a thin snow cover show snow on top.
			int snow = registry.snowMaterial();
			if (snow != 0) {
				for (int y = 0; y < 16; y++) {
					for (int z = 0; z < 16; z++) {
						for (int x = 0; x < 16; x++) {
							int here = (y + PAD) * BORDER * BORDER + (z + PAD) * BORDER + (x + PAD);
							if (occupancy[here] == SMOOTH && occupancy[here + BORDER * BORDER] == COVER) {
								int local = (y << 8) | (z << 4) | x;
								materials[local] = (materials[local] & ~0xFF) | snow;
							}
						}
					}
				}
			}
		}
		return new Result(emitters == null ? new int[0] : Arrays.copyOf(emitters, emitterCount), materials, occupancy);
	}

	private static byte classify(RenderSectionRegion region, BlockPos.MutableBlockPos pos, int x, int y, int z,
			MaterialRegistry registry) {
		BlockState state = region.getBlockState(pos.set(x, y, z));
		if (registry.isSmooth(state.getBlock()) || isDeepSnow(state)) {
			return SMOOTH;
		}
		if (state.getBlock() == Blocks.SNOW) {
			BlockState below = region.getBlockState(pos.set(x, y - 1, z));
			if (registry.isSmooth(below.getBlock()) || isDeepSnow(below)) {
				return COVER;
			}
		}
		if (state.isSolidRender()) {
			return SOLID;
		}
		return state.isCollisionShapeFullBlock(region, pos.set(x, y, z)) ? PIN : OPEN;
	}

	// An emitter enclosed by opaque blocks or more of itself (the inside of a lava lake) cannot
	// light anything; skipping those keeps lava oceans from flooding the light lists.
	private static boolean isExposed(RenderSectionRegion region, BlockPos.MutableBlockPos pos, BlockState state) {
		int x = pos.getX(), y = pos.getY(), z = pos.getZ();
		boolean exposed = false;
		for (net.minecraft.core.Direction direction : net.minecraft.core.Direction.values()) {
			BlockState neighbor = region.getBlockState(pos.set(x + direction.getStepX(), y + direction.getStepY(), z + direction.getStepZ()));
			if (!neighbor.canOcclude() && neighbor.getBlock() != state.getBlock()) {
				exposed = true;
				break;
			}
		}
		pos.set(x, y, z);
		return exposed;
	}

	private static boolean isDeepSnow(BlockState state) {
		return state.getBlock() == Blocks.SNOW && state.getValue(SnowLayerBlock.LAYERS) > MAX_COVER_LAYERS;
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
