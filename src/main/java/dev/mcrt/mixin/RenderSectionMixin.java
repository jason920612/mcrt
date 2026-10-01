package dev.mcrt.mixin;

import dev.mcrt.rt.SectionCapture;
import net.minecraft.client.renderer.chunk.SectionRenderDispatcher;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Tracks which section positions are live, so unloaded/moved sections leave the ray tracer. */
@Mixin(SectionRenderDispatcher.RenderSection.class)
public abstract class RenderSectionMixin {
	@Shadow private volatile long sectionNode;

	@Inject(method = "reset", at = @At("HEAD"))
	private void mcrt$onReset(CallbackInfo ci) {
		SectionCapture.onReset(this.sectionNode);
	}

	@Inject(method = "setSectionNode", at = @At("TAIL"))
	private void mcrt$onAssigned(long sectionNode, CallbackInfo ci) {
		SectionCapture.onAssigned(this.sectionNode);
	}
}
