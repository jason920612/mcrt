package dev.mcrt.rt;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * FFM bindings for mcrt_native. Struct layouts must match native/include/mcrt/api.h,
 * whose static_asserts pin the same offsets.
 */
final class NativeBridge {
	static final int VERTEX_STRIDE = 28;

	static final long FRAME_INPUT_SIZE = 304;
	static final long OFF_COLOR_IMAGE = 0;
	static final long OFF_DEPTH_IMAGE = 8;
	static final long OFF_ATLAS_IMAGE = 16;
	static final long OFF_COLOR_FORMAT = 24;
	static final long OFF_DEPTH_FORMAT = 28;
	static final long OFF_ATLAS_FORMAT = 32;
	static final long OFF_ATLAS_MIP_LEVELS = 36;
	static final long OFF_WIDTH = 40;
	static final long OFF_HEIGHT = 44;
	static final long OFF_ATLAS_WIDTH = 48;
	static final long OFF_ATLAS_HEIGHT = 52;
	static final long OFF_FRAME_INDEX = 56;
	static final long OFF_DEBUG_MODE = 60;
	static final long OFF_CAMERA_BLOCK_POS = 64;
	static final long OFF_FLAGS = 76;
	static final long OFF_CAMERA_OFFSET = 80;
	static final long OFF_VIEW_PROJ = 96;
	static final long OFF_INV_VIEW_PROJ = 160;
	static final long OFF_SUN_DIR = 224;
	static final long OFF_MOON_DIR = 240;
	static final long OFF_SKY_COLOR = 256;
	static final long OFF_TIME = 272;
	static final long OFF_RAIN = 276;
	static final long OFF_PIXEL_SPREAD = 280;
	static final long OFF_CLOUD_HEIGHT = 284;
	static final long OFF_RENDER_SCALE = 288;

	static final long FRAME_OUTPUT_SIZE = 24;
	static final long OFF_OUT_COMMAND_BUFFER = 0;
	static final long OFF_OUT_SEMAPHORE = 8;
	static final long OFF_OUT_SIGNAL_VALUE = 16;

	private final MethodHandle create;
	private final MethodHandle renderFrame;
	private final MethodHandle sectionUpdate;
	private final MethodHandle sectionRemove;
	private final MethodHandle sectionsClear;
	private final MethodHandle getStats;
	private final MethodHandle materialUpload;
	private final MethodHandle destroy;
	private final MethodHandle lastError;

	private NativeBridge(SymbolLookup lib) {
		Linker linker = Linker.nativeLinker();
		this.create = linker.downcallHandle(lib.findOrThrow("mcrt_create"),
			FunctionDescriptor.of(ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT));
		this.renderFrame = linker.downcallHandle(lib.findOrThrow("mcrt_render_frame"),
			FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
		this.sectionUpdate = linker.downcallHandle(lib.findOrThrow("mcrt_section_update"),
			FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT,
				ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));
		this.sectionRemove = linker.downcallHandle(lib.findOrThrow("mcrt_section_remove"),
			FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT));
		this.sectionsClear = linker.downcallHandle(lib.findOrThrow("mcrt_sections_clear"),
			FunctionDescriptor.ofVoid(ADDRESS));
		this.materialUpload = linker.downcallHandle(lib.findOrThrow("mcrt_material_upload"),
			FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_LONG, JAVA_LONG));
		this.getStats = linker.downcallHandle(lib.findOrThrow("mcrt_get_stats"),
			FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));
		this.destroy = linker.downcallHandle(lib.findOrThrow("mcrt_destroy"),
			FunctionDescriptor.ofVoid(ADDRESS));
		this.lastError = linker.downcallHandle(lib.findOrThrow("mcrt_last_error"),
			FunctionDescriptor.of(ADDRESS));
	}

	static NativeBridge load() throws IOException {
		String resource = "/natives/windows-x64/mcrt_native.dll";
		Path dir = Files.createTempDirectory("mcrt");
		Path dll = dir.resolve("mcrt_native.dll");
		try (InputStream in = NativeBridge.class.getResourceAsStream(resource)) {
			if (in == null) {
				throw new IOException("Missing native library resource " + resource);
			}
			Files.copy(in, dll, StandardCopyOption.REPLACE_EXISTING);
		}
		dll.toFile().deleteOnExit();
		dir.toFile().deleteOnExit();
		return new NativeBridge(SymbolLookup.libraryLookup(dll, Arena.global()));
	}

	MemorySegment create(long getInstanceProcAddr, long instance, long physicalDevice, long device, int queueFamily) {
		try {
			return (MemorySegment) create.invokeExact(getInstanceProcAddr, instance, physicalDevice, device, queueFamily);
		} catch (Throwable t) {
			throw new RuntimeException(t);
		}
	}

	int renderFrame(MemorySegment ctx, MemorySegment input, MemorySegment output) {
		try {
			return (int) renderFrame.invokeExact(ctx, input, output);
		} catch (Throwable t) {
			throw new RuntimeException(t);
		}
	}

	void sectionUpdate(MemorySegment ctx, int x, int y, int z,
			MemorySegment solid, int solidVertices, MemorySegment cutout, int cutoutVertices,
			MemorySegment translucent, int translucentVertices, MemorySegment lights, int lightCount,
			MemorySegment blockMaterials, MemorySegment occupancy) {
		try {
			sectionUpdate.invokeExact(ctx, x, y, z, solid, solidVertices, cutout, cutoutVertices, translucent, translucentVertices,
				lights, lightCount, blockMaterials, occupancy);
		} catch (Throwable t) {
			throw new RuntimeException(t);
		}
	}

	void sectionRemove(MemorySegment ctx, int x, int y, int z) {
		try {
			sectionRemove.invokeExact(ctx, x, y, z);
		} catch (Throwable t) {
			throw new RuntimeException(t);
		}
	}

	void sectionsClear(MemorySegment ctx) {
		try {
			sectionsClear.invokeExact(ctx);
		} catch (Throwable t) {
			throw new RuntimeException(t);
		}
	}

	void materialUpload(MemorySegment ctx, int index, int count, int scale, int flags, long file, long fileBytes) {
		try {
			materialUpload.invokeExact(ctx, index, count, scale, flags, file, fileBytes);
		} catch (Throwable t) {
			throw new RuntimeException(t);
		}
	}

	static final long STATS_SIZE = 32;

	void getStats(MemorySegment ctx, MemorySegment stats) {
		try {
			getStats.invokeExact(ctx, stats);
		} catch (Throwable t) {
			throw new RuntimeException(t);
		}
	}

	void destroy(MemorySegment ctx) {
		try {
			destroy.invokeExact(ctx);
		} catch (Throwable t) {
			throw new RuntimeException(t);
		}
	}

	String lastError() {
		try {
			MemorySegment str = (MemorySegment) lastError.invokeExact();
			return str.reinterpret(4096).getString(0);
		} catch (Throwable t) {
			throw new RuntimeException(t);
		}
	}
}
