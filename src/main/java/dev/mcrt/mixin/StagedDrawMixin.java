package dev.mcrt.mixin;

import com.mojang.blaze3d.vertex.MeshData;
import com.mojang.renderpearl.api.pipeline.PrimitiveTopology;
import com.mojang.renderpearl.api.vertex.VertexFormat;
import dev.mcrt.rt.EntityCapture;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Every model mesh Minecraft stages for drawing passes here; entity meshes are copied for ray tracing. */
@Mixin(targets = "net.minecraft.client.renderer.StagedVertexBuffer$Draw")
public abstract class StagedDrawMixin {
	@Shadow @Final private VertexFormat format;
	@Shadow @Final private PrimitiveTopology primitiveTopology;

	@Inject(method = "append", at = @At("HEAD"))
	private void mcrt$captureEntityMesh(MeshData mesh, CallbackInfo ci) {
		if (primitiveTopology == PrimitiveTopology.QUADS && format == com.mojang.blaze3d.vertex.DefaultVertexFormat.ENTITY) {
			EntityCapture.capture(mesh, format.getVertexSize());
		}
	}
}
