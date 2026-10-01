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
	static final int PAD = 3;      // occupancy covers the section plus PAD blocks on every side
	static final int BORDER = 16 + 2 * PAD;

	private static final Map<Block, Integer> LIGHT_COLORS = new ConcurrentHashMap<>();

	/**
	 * @param emitters  each: bits 0-11 local index, 12-15 emission level, 16-31 RGB565 color
	 * @param materials 4096 entries: packed face materials (see MaterialRegistry) in bits 0-23 and the
	 *                  replacement shape in bits 24-31 (SHAPE_*); null if the section has neither
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
					faces |= shapeOf(state) << 24;
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
			// Classify one block wider than we hand out, so the thin-wall rule below sees the
			// neighbors of every cell it decides (keeping neighboring sections consistent).
			final int wide = BORDER + 2;
			byte[] classes = new byte[wide * wide * wide];
			for (int y = -PAD - 1; y < 16 + PAD + 1; y++) {
				for (int z = -PAD - 1; z < 16 + PAD + 1; z++) {
					for (int x = -PAD - 1; x < 16 + PAD + 1; x++) {
						classes[(y + PAD + 1) * wide * wide + (z + PAD + 1) * wide + (x + PAD + 1)] =
							classify(region, pos, baseX + x, baseY + y, baseZ + z, registry);
					}
				}
			}
			occupancy = new byte[BORDER * BORDER * BORDER];
			for (int y = 0; y < BORDER; y++) {
				for (int z = 0; z < BORDER; z++) {
					for (int x = 0; x < BORDER; x++) {
						int w = (y + 1) * wide * wide + (z + 1) * wide + (x + 1);
						byte type = classes[w];
						// A one-block-thick wall (open on both sides across it, continuing along it and
						// up or down) is practically never natural and nearly always built: keep it a
						// cube. Lone bumps in natural terrain fail the continuation tests.
						if (type == SMOOTH && (isWall(classes, w, 1, wide, wide * wide) || isWall(classes, w, wide, 1, wide * wide))) {
							type = SOLID;
						}
						occupancy[y * BORDER * BORDER + z * BORDER + x] = type;
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
			occupancy = appendTerrainInfo(region, pos, occupancy, baseX, baseY, baseZ, registry);
		}
		return new Result(emitters == null ? new int[0] : Arrays.copyOf(emitters, emitterCount), materials, occupancy);
	}

	/**
	 * Terrain mesher input past the occupancy grid (see api.h): top|side material (uint16) of every
	 * smooth block in the padded grid, then the grass color of every padded column (uint32 RGBA8),
	 * so neighboring sections derive identical terrain along their shared faces.
	 */
	private static byte[] appendTerrainInfo(RenderSectionRegion region, BlockPos.MutableBlockPos pos, byte[] occupancy,
			int baseX, int baseY, int baseZ, MaterialRegistry registry) {
		final int cells = BORDER * BORDER * BORDER;
		java.nio.ByteBuffer out = java.nio.ByteBuffer.allocate(cells + cells * 2 + BORDER * BORDER * 4)
			.order(java.nio.ByteOrder.LITTLE_ENDIAN);
		out.put(occupancy);
		int snow = registry.snowMaterial();
		for (int y = 0; y < BORDER; y++) {
			for (int z = 0; z < BORDER; z++) {
				for (int x = 0; x < BORDER; x++) {
					int i = y * BORDER * BORDER + z * BORDER + x;
					int faces = 0;
					if (occupancy[i] == SMOOTH) {
						BlockState state = region.getBlockState(pos.set(baseX + x - PAD, baseY + y - PAD, baseZ + z - PAD));
						faces = isDeepSnow(state) ? snow | (snow << 8) : registry.faces(state.getBlock()) & 0xFFFF;
						if (y + 1 < BORDER && occupancy[i + BORDER * BORDER] == COVER && snow != 0) {
							faces = (faces & ~0xFF) | snow;
						}
					}
					out.putShort((short) faces);
				}
			}
		}
		for (int z = 0; z < BORDER; z++) {
			for (int x = 0; x < BORDER; x++) {
				int rgb = net.minecraft.client.renderer.BiomeColors.getAverageGrassColor(region,
					pos.set(baseX + x - PAD, baseY + 8, baseZ + z - PAD));
				out.putInt(((rgb >> 16) & 0xFF) | (rgb & 0xFF00) | ((rgb & 0xFF) << 16) | 0xFF000000);
			}
		}
		return out.array();
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

	// Blocks re-shaped natively to hide the cube (see native/src/shapes.cpp).
	static final int SHAPE_LEAVES = 1;  // replaced by crossed, alpha-cut foliage cards
	static final int SHAPE_LOG_X = 2;   // logs: replaced by an octagonal trunk along the axis
	static final int SHAPE_LOG_Y = 3;
	static final int SHAPE_LOG_Z = 4;
	static final int SHAPE_HIDDEN = 5;  // pixel-art grass plants: dropped; the terrain's grass cards replace them
	static final int SHAPE_NEEDLES = 6; // conifer leaves: like SHAPE_LEAVES with needle foliage

	private static int shapeOf(BlockState state) {
		Block block = state.getBlock();
		if (block instanceof net.minecraft.world.level.block.LeavesBlock) {
			return block == Blocks.SPRUCE_LEAVES ? SHAPE_NEEDLES : SHAPE_LEAVES;
		}
		if (block == Blocks.SHORT_GRASS || block == Blocks.TALL_GRASS || block == Blocks.FERN || block == Blocks.LARGE_FERN) {
			return SHAPE_HIDDEN;
		}
		if (block instanceof net.minecraft.world.level.block.RotatedPillarBlock) {
			String id = BuiltInRegistries.BLOCK.getKey(block).getPath();
			if (id.endsWith("_log") || id.endsWith("_stem")) {
				return switch (state.getValue(net.minecraft.world.level.block.RotatedPillarBlock.AXIS)) {
					case X -> SHAPE_LOG_X;
					case Z -> SHAPE_LOG_Z;
					default -> SHAPE_LOG_Y;
				};
			}
		}
		return 0;
	}

	// across/along/up: index strides of the axis across the wall, along it, and vertical.
	private static boolean isWall(byte[] classes, int w, int across, int along, int up) {
		if (!isOpen(classes[w - across]) || !isOpen(classes[w + across])) {
			return false;
		}
		boolean continuesAlong = isFilled(classes[w - along]) || isFilled(classes[w + along]);
		boolean continuesVertically = (isFilled(classes[w + up]) && isOpen(classes[w + up - across]) && isOpen(classes[w + up + across]))
			|| (isFilled(classes[w - up]) && isOpen(classes[w - up - across]) && isOpen(classes[w - up + across]));
		return continuesAlong && continuesVertically;
	}

	private static boolean isFilled(byte type) {
		return type == SMOOTH || type == SOLID;
	}

	private static boolean isOpen(byte type) {
		return type == OPEN || type == COVER || type == PIN;
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
