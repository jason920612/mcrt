package dev.mcrt.mixin;

import dev.mcrt.rt.RtRenderer;
import net.minecraft.client.renderer.entity.EntityRenderer;
import net.minecraft.core.BlockPos;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Minecraft lights entities by the light level where they stand, so a mob in a tree's shade is as
 * bright as one in full sun, while the path traced ground around it is in shadow. Under ray tracing
 * an entity the sun (or moon) cannot see gets less sky light, matching its surroundings.
 */
@Mixin(EntityRenderer.class)
public abstract class EntityRendererMixin<T extends Entity> {
	private static final int SHADOWED_SKY_LIGHT_DROP = 5;

	@Inject(method = "getSkyLightLevel", at = @At("RETURN"), cancellable = true)
	private void mcrt$shadowedSkyLight(T entity, BlockPos pos, CallbackInfoReturnable<Integer> cir) {
		int sky = cir.getReturnValue();
		float[] light = RtRenderer.keyLightDirection();
		if (sky == 0 || light == null || !RtRenderer.get().isActive()) {
			return;
		}
		Vec3 from = entity.getBoundingBox().getCenter();
		Vec3 to = from.add(light[0] * 64.0, light[1] * 64.0, light[2] * 64.0);
		HitResult hit = entity.level().clip(new ClipContext(from, to, ClipContext.Block.VISUAL, ClipContext.Fluid.NONE, entity));
		if (hit.getType() != HitResult.Type.MISS) {
			cir.setReturnValue(Math.max(0, sky - SHADOWED_SKY_LIGHT_DROP));
		}
	}
}
