package dev.mcrt.rt;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import dev.mcrt.McrtClient;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.world.level.block.Block;

/**
 * PBR materials (built by tools/materials/build_materials.py) and which block faces use them.
 *
 * <p>Material ids are 1-based; 0 means "use Minecraft's own texture". A block's faces pack into one
 * int: bits 0-7 top, 8-15 side, 16-23 bottom.
 */
public final class MaterialRegistry {
	private static final String ROOT = "/assets/mcrt/materials/";
	static final int FLAG_TINTED = 1;
	/** Pseudo material id: smoothed geometry textured with the block's own atlas sprite. */
	static final int ATLAS_MATERIAL = 254;

	record Material(String name, int scale, boolean tinted) {
	}

	private final List<Material> materials = new ArrayList<>();
	private final Map<Block, Integer> blockFaces = new IdentityHashMap<>();
	private final java.util.Set<Block> smoothBlocks = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
	private int textureSize;
	private int snowMaterial;

	static MaterialRegistry load() {
		MaterialRegistry registry = new MaterialRegistry();
		try (InputStream in = MaterialRegistry.class.getResourceAsStream(ROOT + "materials.json")) {
			if (in == null) {
				McrtClient.LOGGER.warn("MCRT: no material table found; run tools/materials/build_materials.py");
				return registry;
			}
			JsonObject root = JsonParser.parseReader(new InputStreamReader(in, StandardCharsets.UTF_8)).getAsJsonObject();
			registry.textureSize = root.get("size").getAsInt();
			Map<String, Integer> ids = new HashMap<>();
			for (JsonElement element : root.getAsJsonArray("materials")) {
				JsonObject m = element.getAsJsonObject();
				Material material = new Material(m.get("name").getAsString(), m.get("scale").getAsInt(), m.get("tinted").getAsBoolean());
				registry.materials.add(material);
				ids.put(material.name(), registry.materials.size());
			}
			registry.snowMaterial = ids.getOrDefault("snow", 0);
			for (Map.Entry<String, JsonElement> entry : root.getAsJsonObject("blocks").entrySet()) {
				Block block = BuiltInRegistries.BLOCK.getValue(Identifier.parse(entry.getKey()));
				JsonObject faces = entry.getValue().getAsJsonObject();
				int all = id(ids, faces, "all");
				int top = faces.has("top") ? id(ids, faces, "top") : all;
				int side = faces.has("side") ? id(ids, faces, "side") : all;
				int bottom = faces.has("bottom") ? id(ids, faces, "bottom") : all;
				if (faces.has("atlas") && faces.get("atlas").getAsBoolean()) {
					// Smoothed, but keeps Minecraft's own block texture (ores).
					top = side = bottom = ATLAS_MATERIAL;
				}
				registry.blockFaces.put(block, top | (side << 8) | (bottom << 16));
				if (faces.has("smooth") && faces.get("smooth").getAsBoolean()) {
					registry.smoothBlocks.add(block);
				}
			}
		} catch (IOException | RuntimeException e) {
			McrtClient.LOGGER.error("MCRT: failed to read material table", e);
			registry.materials.clear();
			registry.blockFaces.clear();
		}
		return registry;
	}

	private static int id(Map<String, Integer> ids, JsonObject faces, String key) {
		return faces.has(key) ? ids.get(faces.get(key).getAsString()) : 0;
	}

	/** Natural blocks rendered as smoothed terrain. */
	boolean isSmooth(Block block) {
		return smoothBlocks.contains(block);
	}

	/** Material id for snow-covered tops (thin snow layers fold into the surface below), or 0. */
	int snowMaterial() {
		return snowMaterial;
	}

	/** Packed face materials for a block, or 0 when it keeps Minecraft's texture. */
	int faces(Block block) {
		return blockFaces.getOrDefault(block, 0);
	}

	/** Hands every material's .mcm file (block-compressed mip chains) to the native renderer. */
	void upload(NativeBridge bridge, java.lang.foreign.MemorySegment ctx) {
		for (int i = 0; i < materials.size(); i++) {
			Material material = materials.get(i);
			try (InputStream in = MaterialRegistry.class.getResourceAsStream(ROOT + material.name() + ".mcm");
				 java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofConfined()) {
				if (in == null) {
					throw new IOException("missing " + material.name() + ".mcm");
				}
				byte[] bytes = in.readAllBytes();
				java.lang.foreign.MemorySegment file = arena.allocate(bytes.length);
				java.lang.foreign.MemorySegment.copy(bytes, 0, file, java.lang.foreign.ValueLayout.JAVA_BYTE, 0, bytes.length);
				bridge.materialUpload(ctx, i, materials.size(), material.scale(),
					material.tinted() ? FLAG_TINTED : 0, file.address(), bytes.length);
			} catch (IOException e) {
				McrtClient.LOGGER.error("MCRT: failed to load material {}", material.name(), e);
			}
		}
		McrtClient.LOGGER.info("MCRT: uploaded {} materials for {} blocks", materials.size(), blockFaces.size());
	}
}
