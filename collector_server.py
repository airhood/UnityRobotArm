"""
TCP server that receives camera frames + robot state from Unity DataCollector.cs.
Unity is the client; Python is the server.

Wire format per frame (big-endian uint32 length-prefixed):
    [4 bytes: json_len][json_bytes][4 bytes: jpg_len][jpg_bytes]
"""

import io
import json
import logging
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Callable

from PIL import Image

logger = logging.getLogger(__name__)


class DataCollectorServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5006,
        save_dir: str = "data/episodes",
    ):
        self.host = host
        self.port = port
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._sock: socket.socket | None = None
        self._running = False
        self._episode: dict | None = None
        self._callbacks: list[Callable] = []
        self._episode_id = len(list(self.save_dir.glob("episode_*")))
        self._last_unity_episode_id = -1  # last episode_id received from Unity

    # ── Episode control ──────────────────────────────────────────────────────

    def start_episode(self, label: str = ""):
        ep_dir = self.save_dir / f"episode_{self._episode_id:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        self._episode = {
            "id": self._episode_id,
            "label": label,
            "dir": ep_dir,
            "start_time": time.time(),
            "frames": [],
        }
        logger.info(f"Episode {self._episode_id} started: '{label}'")

    def stop_episode(self):
        if not self._episode:
            return
        ep = self._episode
        # Flush states
        with open(ep["dir"] / "states.json", "w") as f:
            json.dump(ep["frames"], f)
        with open(ep["dir"] / "metadata.json", "w") as f:
            json.dump({
                "id": ep["id"],
                "label": ep["label"],
                "start_time": ep["start_time"],
                "end_time": time.time(),
                "frame_count": len(ep["frames"]),
            }, f, indent=2)
        logger.info(f"Episode {ep['id']} saved: {len(ep['frames'])} frames → {ep['dir']}")
        self._episode_id += 1
        self._episode = None

    # ── Server lifecycle ─────────────────────────────────────────────────────

    def listen(self):
        """Start listening in a background thread."""
        self._running = True
        t = threading.Thread(target=self._serve, daemon=True, name="DataCollectorServer")
        t.start()

    def close(self):
        self._running = False
        if self._sock:
            self._sock.close()

    # ── Callbacks ────────────────────────────────────────────────────────────

    def add_callback(self, fn: Callable):
        """Register fn(state: dict, image: PIL.Image) per frame."""
        self._callbacks.append(fn)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _serve(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(1)
        logger.info(f"DataCollectorServer listening on {self.host}:{self.port}")

        while self._running:
            try:
                conn, addr = self._sock.accept()
                logger.info(f"Unity DataCollector connected from {addr}")
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
            except OSError:
                if self._running:
                    raise

    def _handle(self, conn: socket.socket):
        try:
            while self._running:
                json_bytes = self._recv_prefixed(conn)
                if json_bytes is None:
                    break
                img_bytes = self._recv_prefixed(conn)
                if img_bytes is None:
                    break

                state = json.loads(json_bytes)
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                self._on_frame(state, image)
        except Exception as e:
            logger.error(f"Handler error: {e}")
        finally:
            conn.close()
            logger.info("DataCollector Unity client disconnected")

    def _recv_prefixed(self, conn: socket.socket) -> bytes | None:
        header = self._recv_exact(conn, 4)
        if header is None:
            return None
        (length,) = struct.unpack(">I", header)
        return self._recv_exact(conn, length)

    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
        data = b""
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _on_frame(self, state: dict, image: Image.Image):
        # ── Auto-detect episode boundaries ────────────────────────────────
        unity_ep_id = state.get("episode_id", -1)

        if unity_ep_id != self._last_unity_episode_id:
            # end previous episode
            if self._episode is not None:
                self.stop_episode()
            # start new episode (-1 means inter-episode idle)
            if unity_ep_id >= 0:
                self.start_episode(state.get("episode_label", ""))
            self._last_unity_episode_id = unity_ep_id

        # ── Callbacks ─────────────────────────────────────────────────────
        for cb in self._callbacks:
            try:
                cb(state, image)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        # ── Save ──────────────────────────────────────────────────────────
        if not self._episode:
            return

        ep = self._episode
        idx = len(ep["frames"])
        img_path = ep["dir"] / f"frame_{idx:06d}.jpg"
        image.save(img_path, quality=85)

        ep["frames"].append({**state, "frame_idx": idx, "image_path": str(img_path)})

        if idx % 200 == 0 and idx > 0:
            with open(ep["dir"] / "states.json", "w") as f:
                json.dump(ep["frames"], f)
