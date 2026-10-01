package dev.mcrt.mixin;

import com.mojang.blaze3d.vertex.VertexSorting;
import dev.mcrt.rt.SectionCapture;
import net.minecraft.client.renderer.SectionBufferBuilderPack;
import net.minecraft.client.renderer.chunk.RenderSectionRegion;
import net.minecraft.client.renderer.chunk.SectionCompiler;
import net.minecraft.core.SectionPos;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

/** Captures every section mesh Minecraft compiles (on its worker threads) for the ray tracer. */
@Mixin(targets = "net.minecraft.client.renderer.chunk.SectionRenderDispatcher$RenderSection$CompileTask")
public abstract class SectionCompileTaskMixin {
	@Redirect(
		method = "doTask",
		at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/chunk/SectionCompiler;compile(Lnet/minecraft/core/SectionPos;Lnet/minecraft/client/renderer/chunk/RenderSectionRegion;Lcom/mojang/blaze3d/vertex/VertexSorting;Lnet/minecraft/client/renderer/SectionBufferBuilderPack;)Lnet/minecraft/client/renderer/chunk/SectionCompiler$Results;")
	)
	private SectionCompiler.Results mcrt$captureCompiled(SectionCompiler compiler, SectionPos pos, RenderSectionRegion region,
			VertexSorting sorting, SectionBufferBuilderPack buffers) {
		SectionCompiler.Results results = compiler.compile(pos, region, sorting, buffers);
		SectionCapture.onCompiled(pos, region, results);
		return results;
	}
}
