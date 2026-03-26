"""
Unity communication clients.

UnityBridge  : AICommandServer (port 5007) — send/receive robot commands
ImageClient  : ImageServer     (port 5008) — request camera frames
"""

import io
import logging
import socket
import struct
import threading
import time

from PIL import Image

logger = logging.getLogger(__name__)


class UnityBridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 5007):
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._connected = False
        self._buf = ""

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def connect(self, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5.0)
                s.connect((self.host, self.port))
                self._sock = s
                self._connected = True
                logger.info(f"Connected to Unity AICommandServer @ {self.host}:{self.port}")
                return True
            except (ConnectionRefusedError, OSError):
                time.sleep(0.5)
        logger.error(f"Could not connect to Unity after {timeout}s")
        return False

    def disconnect(self):
        self._connected = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass

    # ── Commands ─────────────────────────────────────────────────────────────

    def move_to(self, x: float, y: float, z: float) -> bool:
        """Move end-effector target to absolute world position."""
        return self._cmd(f"MOVE_TO {x:.4f},{y:.4f},{z:.4f}") == "OK"

    def move_rel(self, dx: float, dy: float, dz: float) -> bool:
        """Move target by relative offset."""
        return self._cmd(f"MOVE_REL {dx:.4f},{dy:.4f},{dz:.4f}") == "OK"

    def gripper_open(self) -> bool:
        return self._cmd("GRIPPER open") == "OK"

    def gripper_close(self) -> bool:
        return self._cmd("GRIPPER close") == "OK"

    def get_state(self) -> dict | None:
        """
        Returns:
            {
              'joint_angles': [j0..j4],   # degrees
              'ee_position':  [x,y,z],    # Unity world coords
              'target_position': [x,y,z],
            }
        """
        resp = self._cmd("GET_STATE")
        if resp and resp.startswith("STATE "):
            return _parse_state(resp[6:])
        return None

    # ── Context manager ──────────────────────────────────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _cmd(self, msg: str) -> str | None:
        with self._lock:
            if not self._connected:
                return None
            try:
                self._sock.sendall((msg + "\n").encode())
                return self._recv_line()
            except Exception as e:
                logger.warning(f"Command failed ({msg!r}): {e}")
                self._connected = False
                return None

    def _recv_line(self) -> str:
        while "\n" not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("Socket closed")
            self._buf += chunk.decode()
        line, self._buf = self._buf.split("\n", 1)
        return line.strip()


def _parse_state(data: str) -> dict:
    """Parse 'j0,...,j4,ex,ey,ez,tx,ty,tz' into a structured dict."""
    vals = [float(v) for v in data.split(",")]
    return {
        "joint_angles": vals[:5],
        "ee_position": vals[5:8],
        "target_position": vals[8:11],
    }


class ImageClient:
    """Requests camera frames from Unity ImageServer (port 5008)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5008):
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._connected = False

    def connect(self, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5.0)
                s.connect((self.host, self.port))
                self._sock = s
                self._connected = True
                logger.info(f"Connected to Unity ImageServer @ {self.host}:{self.port}")
                return True
            except (ConnectionRefusedError, OSError):
                time.sleep(0.5)
        logger.error(f"Could not connect to ImageServer after {timeout}s")
        return False

    def disconnect(self):
        self._connected = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass

    def get_image(self) -> Image.Image | None:
        """Send GET request to Unity → receive JPEG → return PIL Image."""
        with self._lock:
            if not self._connected:
                return None
            try:
                self._sock.sendall(b"GET\n")
                # read 4-byte length header
                header = self._recv_exact(4)
                if header is None:
                    return None
                (length,) = struct.unpack(">I", header)
                # read JPEG payload
                jpg_bytes = self._recv_exact(length)
                if jpg_bytes is None:
                    return None
                return Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
            except Exception as e:
                logger.warning(f"get_image failed: {e}")
                self._connected = False
                return None

    def _recv_exact(self, n: int) -> bytes | None:
        data = b""
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
