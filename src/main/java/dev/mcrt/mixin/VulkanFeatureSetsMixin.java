package dev.mcrt.mixin;

import com.mojang.renderpearl.backend.vulkan.VulkanFeatureSets;
import com.mojang.renderpearl.backend.vulkan.init.FeatureSet;
import dev.mcrt.rt.RayTracingFeatureSets;
import java.util.Set;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Makes hardware ray tracing a hard requirement of the Vulkan device, so the backend
 * enables the extensions and features our native renderer relies on.
 */
@Mixin(VulkanFeatureSets.class)
public class VulkanFeatureSetsMixin {
	@Inject(method = "requiredFeatureSets", at = @At("RETURN"))
	private static void mcrt$requireRayTracing(CallbackInfoReturnable<Set<FeatureSet>> cir) {
		cir.getReturnValue().add(RayTracingFeatureSets.RAY_TRACING);
	}
}
