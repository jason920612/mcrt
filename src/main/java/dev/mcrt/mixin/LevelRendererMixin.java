package dev.mcrt.mixin;

import com.mojang.blaze3d.framegraph.FrameGraphBuilder;
import com.mojang.blaze3d.framegraph.FramePass;
import com.mojang.renderpearl.api.buffers.GpuBufferSlice;
import com.mojang.renderpearl.api.commands.RenderPass;
import com.mojang.renderpearl.api.textures.GpuSampler;
import com.mojang.renderpearl.api.textures.GpuTextureView;
import dev.mcrt.rt.RtRenderer;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.client.renderer.LevelRenderer;
import net.minecraft.client.renderer.LevelTargetBundle;
import net.minecraft.client.renderer.chunk.ChunkSectionLayerGroup;
import net.minecraft.client.renderer.chunk.ChunkSectionsToRender;
import net.minecraft.client.renderer.feature.FeatureRenderDispatcher;
import net.minecraft.client.renderer.oit.OitRenderPassProvider;
import net.minecraft.client.renderer.oit.OitStage;
import net.minecraft.client.renderer.state.level.LevelRenderState;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Replaces terrain and sky rasterization with the path traced world pass. Everything else
 * (entities, block entities, particles, outlines) is still rasterized by Minecraft on top.
 */
@Mixin(LevelRenderer.class)
public abstract class LevelRendererMixin {
	@Shadow @Final private LevelTargetBundle targets;
	@Shadow @Final private LevelRenderState levelRenderState;
	@Shadow @Final private GameRenderer gameRenderer;

	@Inject(method = "addMainPass", at = @At("HEAD"))
	private void mcrt$addPathTracePass(FrameGraphBuilder frame, FeatureRenderDispatcher.PreparedFrame featureFrame,
			GpuBufferSlice terrainFog, ChunkSectionsToRender chunkSectionsToRender, boolean consistentDepthRequired,
			CallbackInfo ci) {
		RtRenderer renderer = RtRenderer.get();
		if (!renderer.isActive()) {
			return;
		}
		// Reads and writes the main target, so the frame graph orders it right before "main".
		FramePass pass = frame.addPass("mcrt_path_trace");
		this.targets.main = pass.readsAndWrites(this.targets.main);
		pass.executes(() -> renderer.renderWorld(this.levelRenderState, this.gameRenderer.mainRenderTarget()));
	}

	/**
	 * Keeps addSkyPass running (it creates the SkyRenderer, without which Minecraft never extracts
	 * the sky state we read sun/moon angles from) but drops the sky draw itself.
	 */
	@Redirect(
		method = "addSkyPass",
		at = @At(value = "INVOKE", target = "Lcom/mojang/blaze3d/framegraph/FramePass;executes(Ljava/lang/Runnable;)V")
	)
	private void mcrt$skipSkyDraw(FramePass pass, Runnable draw) {
		pass.executes(RtRenderer.get().isActive() ? () -> { } : draw);
	}

	@Redirect(
		method = {"executeSolid", "executeClassicTransparency"},
		at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/chunk/ChunkSectionsToRender;renderGroup(Lnet/minecraft/client/renderer/chunk/ChunkSectionLayerGroup;Lcom/mojang/renderpearl/api/commands/RenderPass;Lcom/mojang/renderpearl/api/textures/GpuSampler;Lcom/mojang/renderpearl/api/textures/GpuTextureView;Z)V")
	)
	private void mcrt$skipTerrain(ChunkSectionsToRender sections, ChunkSectionLayerGroup group, RenderPass renderPass,
			GpuSampler sampler, GpuTextureView atlas, boolean wireframe) {
		if (!RtRenderer.get().isActive()) {
			sections.renderGroup(group, renderPass, sampler, atlas, wireframe);
		}
	}

	@Redirect(
		method = "executeOit",
		at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/chunk/ChunkSectionsToRender;renderOit(Lcom/mojang/renderpearl/api/textures/GpuSampler;Lnet/minecraft/client/renderer/oit/OitStage;Lnet/minecraft/client/renderer/oit/OitRenderPassProvider$Parameters;Lcom/mojang/renderpearl/api/textures/GpuTextureView;Lcom/mojang/renderpearl/api/textures/GpuTextureView;)V")
	)
	private void mcrt$skipOitTerrain(ChunkSectionsToRender sections, GpuSampler sampler, OitStage stage,
			OitRenderPassProvider.Parameters parameters, GpuTextureView atlas, GpuTextureView lightmap) {
		if (!RtRenderer.get().isActive()) {
			sections.renderOit(sampler, stage, parameters, atlas, lightmap);
		}
	}

	@Inject(method = "invalidateCompiledGeometry", at = @At("HEAD"))
	private void mcrt$onInvalidateGeometry(CallbackInfo ci) {
		RtRenderer.get().onGeometryInvalidated();
	}

	@Inject(method = "resetLevelRenderData", at = @At("HEAD"))
	private void mcrt$onResetLevel(CallbackInfo ci) {
		RtRenderer.get().clearSections();
	}
}
