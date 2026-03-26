# ── Network ports ──────────────────────────────────────────────────────────
IK_SERVER_HOST = "127.0.0.1"
IK_SERVER_PORT = 5005          # manipulator-sim IK server (existing)

UNITY_COMMAND_HOST = "127.0.0.1"
UNITY_COMMAND_PORT = 5007      # AICommandServer.cs in Unity

DATA_COLLECTOR_HOST = "127.0.0.1"
DATA_COLLECTOR_PORT = 5006     # DataCollectorServer (Python) ← Unity connects

IMAGE_SERVER_PORT   = 5008     # ImageServer.cs in Unity (camera frame requests during inference)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = "data/episodes"
MODEL_DIR = "model/checkpoints"

# ── Model hyperparameters ──────────────────────────────────────────────────
IMAGE_SIZE = 224
JOINT_DIM = 5
TEXT_EMBED_DIM = 384    # all-MiniLM-L6-v2 output dim
IMAGE_EMBED_DIM = 512
HIDDEN_DIM = 256
ACTION_DIM = 3          # predicted delta (dx, dy, dz)

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100

# ── Robot workspace (Unity world coords, meters) ───────────────────────────
WORKSPACE = {
    "x": (-0.3, 0.3),
    "y": (0.05, 0.4),
    "z": (-0.3, 0.3),
}
