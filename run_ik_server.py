import time
import numpy as np

import config
from utils.ik_solver import IKSolver
from utils.ik_server import IKServer


def start() -> IKServer:
    ik = IKSolver(
        link_lengths={
            "l0":  0.036,
            "l1":  0.0405,
            "l2a": 0.128,
            "l2b": 0.024,
            "l3":  0.124,
            "l4":  0.06404,
            "l5":  0.13906,
        },
        joint_limits=[(-np.pi, np.pi)] * 5,
    )
    server = IKServer(host=config.IK_SERVER_HOST, port=config.IK_SERVER_PORT, ik_solver=ik)
    server.startServer()
    return server


if __name__ == "__main__":
    server = start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.close()
        print("IK server stopped.")
