package dev.mcrt.mixin;

import dev.mcrt.rt.OffscreenSectionScheduler;
import dev.mcrt.rt.RtRenderer;
import net.minecraft.client.Camera;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.SectionUpdateTracker;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.LevelRenderer;
import net.minecraft.client.renderer.extract.LevelExtractor;
import net.minecraft.client.renderer.state.level.LevelRenderState;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Minecraft only compiles sections that are inside the view frustum and pass occlusion culling.
 * A ray tracer also needs what is behind the camera (shadows, bounce light), so dirty sections
 * outside that set are scheduled too, after Minecraft has scheduled the visible ones.
 */
@Mixin(LevelExtractor.class)
public abstract class LevelExtractorMixin {
	@Shadow private ClientLevel level;
	@Shadow private SectionUpdateTracker sectionUpdateTracker;
	@Shadow @Final private LevelRenderState levelRenderState;
	@Shadow @Final private LevelRenderer levelRenderer;

	@Unique private final OffscreenSectionScheduler mcrt$offscreen = new OffscreenSectionScheduler();

	@Inject(
		method = "extract",
		at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/extract/LevelExtractor;extractVisibleEntities(Lnet/minecraft/client/Camera;Lnet/minecraft/client/renderer/culling/Frustum;Lnet/minecraft/client/DeltaTracker;Lnet/minecraft/client/renderer/state/level/LevelRenderState;)V")
	)
	private void mcrt$scheduleOffscreenSections(DeltaTracker deltaTracker, Camera camera, float partialTicks, CallbackInfo ci) {
		if (RtRenderer.get().isActive() && this.level != null && this.sectionUpdateTracker != null) {
			mcrt$offscreen.schedule(this.levelRenderer.viewArea(), this.level, this.sectionUpdateTracker,
				camera.blockPosition(), this.levelRenderState.sectionUpdateRenderStates);
		}
	}
}
