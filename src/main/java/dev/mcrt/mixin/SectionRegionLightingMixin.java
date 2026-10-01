package dev.mcrt.mixin;

import dev.mcrt.rt.RtRenderer;
import net.minecraft.client.renderer.chunk.RenderSectionRegion;
import net.minecraft.world.level.CardinalLighting;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Section meshes normally bake per-face shading (top 1.0, sides 0.6-0.8, bottom 0.5) into vertex
 * colors. The path tracer lights faces itself, so with it active the bake becomes a no-op and
 * vertex colors carry only the biome tint.
 */
@Mixin(RenderSectionRegion.class)
public class SectionRegionLightingMixin {
	@Unique private static final CardinalLighting MCRT_UNIFORM = new CardinalLighting(1f, 1f, 1f, 1f, 1f, 1f);

	@Inject(method = "cardinalLighting", at = @At("HEAD"), cancellable = true)
	private void mcrt$noBakedFaceShading(CallbackInfoReturnable<CardinalLighting> cir) {
		if (RtRenderer.get().isActive()) {
			cir.setReturnValue(MCRT_UNIFORM);
		}
	}
}
