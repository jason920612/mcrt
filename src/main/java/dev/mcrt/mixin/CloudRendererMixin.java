package dev.mcrt.mixin;

import com.mojang.renderpearl.api.commands.RenderPass;
import com.mojang.renderpearl.api.textures.GpuTextureView;
import dev.mcrt.rt.RtRenderer;
import net.minecraft.client.CloudStatus;
import net.minecraft.client.renderer.CloudRenderer;
import net.minecraft.client.renderer.oit.OitRenderPassProvider;
import net.minecraft.client.renderer.oit.OitStage;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** The path tracer draws volumetric clouds itself; Minecraft's flat cloud layer would sit on top. */
@Mixin(CloudRenderer.class)
public class CloudRendererMixin {
	@Inject(method = "render", at = @At("HEAD"), cancellable = true)
	private void mcrt$skipClouds(CloudStatus status, RenderPass renderPass, CallbackInfo ci) {
		if (RtRenderer.get().isActive()) {
			ci.cancel();
		}
	}

	@Inject(method = "renderOit", at = @At("HEAD"), cancellable = true)
	private void mcrt$skipOitClouds(CloudStatus status, OitStage stage, GpuTextureView depth,
			OitRenderPassProvider.Parameters parameters, CallbackInfo ci) {
		if (RtRenderer.get().isActive()) {
			ci.cancel();
		}
	}
}
