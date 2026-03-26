import socket
import threading

from utils.coordinate_transform import unity_to_rh


class IKServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 5005, enable_log: bool = True, ik_solver=None):
        self.host = host
        self.port = port
        self.enable_log = enable_log
        self.ik_solver = ik_solver
        self.server = None
        self.conn = None
        self.running = False
        self.accept_thread = None

    def log(self, *args):
        if self.enable_log:
            print("[IKServer]", *args)

    def startServer(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)

        self.running = True
        self.accept_thread = threading.Thread(target=self._acceptLoop, daemon=True)
        self.accept_thread.start()

        self.log(f"Server started on {self.host}:{self.port}")

    def _acceptLoop(self):
        connection_count = 0

        while self.running:
            try:
                if connection_count == 0:
                    self.log("Waiting for Unity connection...")
                else:
                    self.log("Waiting for next connection...")

                conn, addr = self.server.accept()
                connection_count += 1

                if self.conn:
                    try:
                        self.conn.close()
                        self.log("Previous connection closed")
                    except:
                        pass

                self.conn = conn
                self.log(f"Connected: {addr} (connection #{connection_count})")

                recv_thread = threading.Thread(target=self._recvLoop, args=(conn,), daemon=True)
                recv_thread.start()

            except Exception as e:
                if self.running:
                    self.log(f"Accept error: {e}")
                break

    def _recvLoop(self, conn: socket.socket):
        """
        Receive format: "x,y,z\n"
        Send format:    "j1,j2,j3,j4,j5\n" (degrees) or "IK_FAIL\n"
        """
        buffer = ""
        while self.running:
            try:
                data = conn.recv(1024).decode()
                if not data:
                    self.log("Connection closed by Unity")
                    break

                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    self._handleTargetPose(conn, line)

            except (ConnectionResetError, BrokenPipeError, OSError):
                self.log("Recv connection lost")
                break

    def _handleTargetPose(self, conn: socket.socket, line: str):
        try:
            parts = line.split(",")
            if len(parts) != 3:
                self.log(f"Invalid format: {line}")
                return

            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            self.log(f"Target (Unity): x={x}, y={y}, z={z}")

            # Unity left-handed → right-handed coordinate conversion
            x, y, z = unity_to_rh(x, y, z)

            if self.ik_solver is None:
                self.log("No IK solver attached")
                conn.sendall(b"IK_FAIL\n")
                return

            angles = self.ik_solver.solve(x, y, z)

            if angles is not None:
                response = ",".join(f"{a:.2f}" for a in angles) + "\n"
                self.log(f"IK solved: {[f'{a:.1f}' for a in angles]}")
            else:
                response = "IK_FAIL\n"
                self.log(f"IK failed for ({x}, {y}, {z})")

            conn.sendall(response.encode())

        except ValueError as e:
            self.log(f"Parse error: {e}")

    def sendServoPositions(self, angles: list):
        """Manually send joint angles."""
        if self.conn is None:
            self.log("No Unity connection.")
            return False
        try:
            message = ",".join(map(str, angles)) + "\n"
            self.conn.sendall(message.encode())
            return True
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            self.log(f"Send failed: {e}")
            self.conn = None
            return False

    def close(self):
        self.running = False

        if self.conn:
            try:
                self.conn.close()
            except:
                pass
            self.conn = None

        if self.server:
            try:
                self.server.close()
            except:
                pass
            self.server = None

        self.log("Server closed")
