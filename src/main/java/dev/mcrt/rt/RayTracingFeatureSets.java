package dev.mcrt.rt;

import static com.mojang.renderpearl.backend.vulkan.VulkanFeatureSets.VK10_FEATURES_STRUCT;
import static com.mojang.renderpearl.backend.vulkan.VulkanFeatureSets.VK12_FEATURES_STRUCT;

import com.mojang.renderpearl.backend.vulkan.init.FeatureSet;
import com.mojang.renderpearl.backend.vulkan.init.VulkanFeature;
import com.mojang.renderpearl.backend.vulkan.init.VulkanPNextStruct;
import java.util.Set;
import org.lwjgl.vulkan.VkPhysicalDeviceAccelerationStructureFeaturesKHR;
import org.lwjgl.vulkan.VkPhysicalDeviceRayQueryFeaturesKHR;
import org.lwjgl.vulkan.VkPhysicalDeviceRayTracingPipelineFeaturesKHR;

public final class RayTracingFeatureSets {
	private static final VulkanPNextStruct ACCELERATION_STRUCTURE_FEATURES =
		new VulkanPNextStruct(VkPhysicalDeviceAccelerationStructureFeaturesKHR.class);
	private static final VulkanPNextStruct RAY_TRACING_PIPELINE_FEATURES =
		new VulkanPNextStruct(VkPhysicalDeviceRayTracingPipelineFeaturesKHR.class);
	private static final VulkanPNextStruct RAY_QUERY_FEATURES =
		new VulkanPNextStruct(VkPhysicalDeviceRayQueryFeaturesKHR.class);

	public static final FeatureSet RAY_TRACING = new FeatureSet(
		"MCRT hardware ray tracing",
		Set.of(
			"VK_KHR_acceleration_structure",
			"VK_KHR_ray_tracing_pipeline",
			"VK_KHR_ray_query",
			"VK_KHR_deferred_host_operations"
		),
		Set.of(
			new VulkanFeature(ACCELERATION_STRUCTURE_FEATURES, "accelerationStructure"),
			new VulkanFeature(RAY_TRACING_PIPELINE_FEATURES, "rayTracingPipeline"),
			new VulkanFeature(RAY_QUERY_FEATURES, "rayQuery"),
			new VulkanFeature(VK10_FEATURES_STRUCT, "shaderInt64"),
			new VulkanFeature(VK12_FEATURES_STRUCT, "bufferDeviceAddress"),
			new VulkanFeature(VK12_FEATURES_STRUCT, "scalarBlockLayout"),
			new VulkanFeature(VK12_FEATURES_STRUCT, "descriptorIndexing"),
			new VulkanFeature(VK12_FEATURES_STRUCT, "runtimeDescriptorArray"),
			new VulkanFeature(VK12_FEATURES_STRUCT, "descriptorBindingPartiallyBound"),
			new VulkanFeature(VK12_FEATURES_STRUCT, "descriptorBindingVariableDescriptorCount"),
			new VulkanFeature(VK12_FEATURES_STRUCT, "shaderSampledImageArrayNonUniformIndexing"),
			new VulkanFeature(VK12_FEATURES_STRUCT, "shaderStorageBufferArrayNonUniformIndexing")
		)
	);

	private RayTracingFeatureSets() {
	}
}
