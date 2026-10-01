package dev.mcrt.mixin;

import dev.mcrt.rt.RtRenderer;
import net.minecraft.client.renderer.GameRenderer;
import org.joml.Matrix4f;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.ModifyArg;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(GameRenderer.class)
public class GameRendererMixin {
	/** Captures the level projection after view bobbing / nausea are applied. */
	@ModifyArg(
		method = "renderLevel",
		at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/ProjectionMatrixBuffer;getBuffer(Lorg/joml/Matrix4f;)Lcom/mojang/renderpearl/api/buffers/GpuBufferSlice;")
	)
	private Matrix4f mcrt$captureLevelProjection(Matrix4f projection) {
		RtRenderer.get().setLevelProjection(projection);
		return projection;
	}

	@Inject(method = "close", at = @At("HEAD"))
	private void mcrt$beforeClose(CallbackInfo ci) {
		RtRenderer.get().shutdown();
	}
}
