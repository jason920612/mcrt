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
	/** Dev aid: -Dmcrt.devCommands="cmd1;cmd2" runs server commands once after joining a singleplayer world. */
	private static final String DEV_COMMANDS = System.getProperty("mcrt.devCommands");
	/** Dev aid: -Dmcrt.devSpin=1.5 turns the player this many degrees per tick, to test motion. */
	private static final float DEV_SPIN = Float.parseFloat(System.getProperty("mcrt.devSpin", "0"));
	/** Dev aid: -Dmcrt.devFly=1.0 flies the player forward this many blocks per tick, to test streaming. */
	private static final float DEV_FLY = Float.parseFloat(System.getProperty("mcrt.devFly", "0"));
	private IntegratedServer devSetupAppliedTo;
	private int devSetupDelay;

	@Override
	public void onInitializeClient() {
		LOGGER.info("MCRT loaded; ray tracing feature sets will be requested from the Vulkan backend");
		if (DEV_TIME != null || DEV_COMMANDS != null) {
			ClientTickEvents.END_CLIENT_TICK.register(this::applyDevSetup);
		}
		if (DEV_FLY != 0f) {
			ClientTickEvents.END_CLIENT_TICK.register(minecraft -> {
				if (minecraft.player != null && minecraft.player.isSpectator()) {
					float yaw = (float) Math.toRadians(minecraft.player.getYRot());
					minecraft.player.setDeltaMovement(-Math.sin(yaw) * DEV_FLY, 0.0, Math.cos(yaw) * DEV_FLY);
				}
			});
		}
		if (DEV_SPIN != 0f) {
			ClientTickEvents.END_CLIENT_TICK.register(minecraft -> {
				if (minecraft.player != null) {
					minecraft.player.setYRot(minecraft.player.getYRot() + DEV_SPIN);
				}
			});
		}
	}

	private void applyDevSetup(Minecraft minecraft) {
		IntegratedServer server = minecraft.getSingleplayerServer();
		if (server == null || server == devSetupAppliedTo || minecraft.level == null || minecraft.player == null) {
			devSetupDelay = 0;
			return;
		}
		// Give the server a moment to place the player before teleporting it around.
		if (++devSetupDelay < 40) {
			return;
		}
		devSetupAppliedTo = server;
		server.execute(() -> {
			var source = server.createCommandSourceStack().withSuppressedOutput();
			if (DEV_TIME != null) {
				server.getCommands().performPrefixedCommand(source, "gamerule advance_time false");
				server.getCommands().performPrefixedCommand(source, "time set " + DEV_TIME);
				server.getCommands().performPrefixedCommand(source, "weather clear");
			}
			if (DEV_COMMANDS != null) {
				for (String command : DEV_COMMANDS.split(";")) {
					server.getCommands().performPrefixedCommand(source, command.trim());
				}
			}
		});
		LOGGER.info("MCRT dev: time={} commands={}", DEV_TIME, DEV_COMMANDS);
	}
}
