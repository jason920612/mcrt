package dev.mcrt;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.client.server.IntegratedServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class McrtClient implements ClientModInitializer {
	public static final String MOD_ID = "mcrt";
	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	/** Dev aid: -Dmcrt.devTime=6000 pins singleplayer worlds to that time of day for repeatable tests. */
	private static final String DEV_TIME = System.getProperty("mcrt.devTime");
	private IntegratedServer devTimeAppliedTo;

	@Override
	public void onInitializeClient() {
		LOGGER.info("MCRT loaded; ray tracing feature sets will be requested from the Vulkan backend");
		if (DEV_TIME != null) {
			ClientTickEvents.END_CLIENT_TICK.register(this::applyDevTime);
		}
	}

	private void applyDevTime(Minecraft minecraft) {
		IntegratedServer server = minecraft.getSingleplayerServer();
		if (server == null || server == devTimeAppliedTo || minecraft.level == null) {
			return;
		}
		devTimeAppliedTo = server;
		server.execute(() -> {
			var source = server.createCommandSourceStack().withSuppressedOutput();
			server.getCommands().performPrefixedCommand(source, "gamerule advance_time false");
			server.getCommands().performPrefixedCommand(source, "time set " + DEV_TIME);
			server.getCommands().performPrefixedCommand(source, "weather clear");
		});
		LOGGER.info("MCRT dev: pinned time of day to {}", DEV_TIME);
	}
}
