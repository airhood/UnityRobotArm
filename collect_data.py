"""
Automated data collection script.

Unity's PickPlaceDemo.cs manages episode boundaries (start/end) automatically.
Python just starts the servers and waits.
Episode labels and boundaries are sent from Unity via DataCollector as JSON.

Usage:
    python collect_data.py
    python collect_data.py --group my_group
    python collect_data.py --save-dir data/episodes --group my_group
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import config
from collector_server import DataCollectorServer
from run_ik_server import start as start_ik_server
from utils.log_setup import setup as setup_logging

logger = setup_logging("collect")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host",     default=config.DATA_COLLECTOR_HOST)
    ap.add_argument("--port",     type=int, default=config.DATA_COLLECTOR_PORT)
    ap.add_argument("--save-dir", default=config.DATA_DIR)
    ap.add_argument("--group",    default=config.DEFAULT_GROUP, help="episode group name (default: '%(default)s')")
    args = ap.parse_args()

    args.save_dir = str(Path(args.save_dir) / args.group)

    ik_server = start_ik_server()

    server = DataCollectorServer(host=args.host, port=args.port, save_dir=args.save_dir)

    ep_count   = [0]
    frame_count = [0]

    def on_frame(state, image):
        frame_count[0] += 1
        ep_id = state.get("episode_id", -1)
        if ep_id > ep_count[0]:
            ep_count[0] = ep_id
        if frame_count[0] % 200 == 0:
            label = state.get("episode_label", "")
            phase = state.get("phase", "")
            logger.info(
                f"  episode={ep_id}  phase={phase:<12}  "
                f"total_frames={frame_count[0]}  label='{label}'"
            )

    server.add_callback(on_frame)
    server.listen()

    print(f"\nIK server started on port {config.IK_SERVER_PORT}")
    print(f"Data collection server started on port {args.port}")
    print(f"Saving to group: '{args.group}' → {args.save_dir}")
    print("Unity PickPlaceDemo will manage episodes automatically.")
    print("Press Ctrl+C to stop.\n")

    def _stop(sig, frame):
        print(f"\nStopping... episodes={ep_count[0]+1}  frames={frame_count[0]}")
        if server._episode:
            server.stop_episode()
        server.close()
        ik_server.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
