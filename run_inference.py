"""
Text-command inference interface using the trained CLIP-based model.

Usage:
    python run_inference.py

Flow:
    1. User enters a text command
    2. Receive camera image + robot state from Unity
    3. Predict Δposition via RobotInference.predict()
    4. Send move command to Unity
    5. Repeat 2~4 until goal reached or max steps
"""

import logging
import math
import time

import config
from model.inference import RobotInference
from utils.unity_bridge import ImageClient, UnityBridge

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("server")

# ── Inference loop settings ───────────────────────────────────────────────
MAX_STEPS      = 50     # max steps per command
STEP_INTERVAL  = 0.2    # seconds between steps
DONE_THRESHOLD = 0.01   # predicted delta magnitude below this → done (m)


def main():
    # ── Load model ────────────────────────────────────────────────────────
    print("Loading model...")
    inference = RobotInference()

    # ── Connect to Unity ──────────────────────────────────────────────────
    bridge     = UnityBridge(config.UNITY_COMMAND_HOST, config.UNITY_COMMAND_PORT)
    img_client = ImageClient(config.UNITY_COMMAND_HOST, config.IMAGE_SERVER_PORT)

    print("Connecting to Unity...")
    if not bridge.connect(timeout=15):
        print("ERROR: Could not connect to AICommandServer. Is Unity running?")
        return
    if not img_client.connect(timeout=15):
        print("ERROR: Could not connect to ImageServer. Is Unity running?")
        bridge.disconnect()
        return

    print("Connected!\n")
    print("Enter a command (e.g. 'pick up the red cube'). Type 'quit' to exit.\n")

    while True:
        try:
            command = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not command:
            continue
        if command.lower() in ("quit", "exit", "q"):
            break

        run_inference_loop(command, inference, bridge, img_client)

    bridge.disconnect()
    img_client.disconnect()
    print("Exiting.")


def run_inference_loop(
    command: str,
    inference: RobotInference,
    bridge: UnityBridge,
    img_client: ImageClient,
):
    """Run the inference loop for a single command."""
    logger.info(f"Command: '{command}'")

    for step in range(MAX_STEPS):
        # 1. get current state
        state = bridge.get_state()
        image = img_client.get_image()

        if state is None or image is None:
            logger.warning("Failed to receive state/image from Unity.")
            break

        # 2. run inference
        result = inference.predict(
            text=command,
            image=image,
            joint_angles=state["joint_angles"],
            ee_position=state["ee_position"],
        )

        delta   = result["delta_position"]   # [dx, dy, dz]
        gripper = result["gripper_open"]

        magnitude = math.sqrt(sum(d * d for d in delta))
        logger.info(
            f"  step={step+1:2d}  "
            f"delta=({delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f})  "
            f"|Δ|={magnitude:.4f}m  gripper={'open' if gripper else 'close'}"
        )

        # 3. check done
        if magnitude < DONE_THRESHOLD:
            logger.info("Done (delta below threshold)")
            break

        # 4. send command
        bridge.move_rel(delta[0], delta[1], delta[2])

        if gripper:
            bridge.gripper_open()
        else:
            bridge.gripper_close()

        time.sleep(STEP_INTERVAL)

    else:
        logger.info(f"Max steps ({MAX_STEPS}) reached")


if __name__ == "__main__":
    main()
