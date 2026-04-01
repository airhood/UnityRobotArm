"""
Inference pipeline for the CLIP-based robot arm model.
Used by server.py for visual grounding when a checkpoint is available.
"""

import logging
from pathlib import Path

import clip
import torch
from PIL import Image

import config
from model.architecture import RobotArmModel
from model.dataset import _EE_CENTER, _EE_SCALE, _JOINT_SCALE

logger = logging.getLogger(__name__)


class RobotInference:
    """Load a trained checkpoint and predict target delta from text + image."""

    def __init__(self, checkpoint: str | None = None, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.model = RobotArmModel(
            clip_dim=512,
            hidden_dim=config.HIDDEN_DIM,
            action_dim=config.ACTION_DIM,
        ).to(self.device)

        ckpt_path = Path(checkpoint or (Path(config.MODEL_DIR) / "best_model.pt"))
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded checkpoint: {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
        else:
            logger.warning(f"No checkpoint at {ckpt_path} — using untrained model")

        self.model.eval()

    def predict(
        self,
        text: str,
        image: Image.Image,
        joint_angles: list[float],
        ee_position: list[float],
    ) -> dict:
        """
        Returns:
            {
              "delta_position":  [dx, dy, dz],   # meters, Unity coords
              "target_position": [x, y, z],      # absolute Unity coords
              "gripper_open":    bool,
            }
        """
        with torch.no_grad():
            img_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            tokens = clip.tokenize([text], truncate=True).to(self.device)

            img_feat  = self.clip_model.encode_image(img_tensor).float()
            text_feat = self.clip_model.encode_text(tokens).float()
            img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            ee_raw = torch.tensor(ee_position, dtype=torch.float32)
            ja = (torch.tensor(joint_angles, dtype=torch.float32) / _JOINT_SCALE).unsqueeze(0).to(self.device)
            ee = ((ee_raw - _EE_CENTER) / _EE_SCALE).unsqueeze(0).to(self.device)

            delta_norm, gripper_logit = self.model(img_feat, text_feat, ja, ee)
            # denormalize: delta in normalized space → meters
            delta_norm = delta_norm.squeeze(0).cpu()
            target_norm = ee.squeeze(0).cpu() + delta_norm
            target = (target_norm * _EE_SCALE + _EE_CENTER).tolist()
            delta = [target[i] - ee_position[i] for i in range(3)]
            gripper_open = torch.sigmoid(gripper_logit).item() > 0.5

        return {
            "delta_position":  delta,
            "target_position": target,
            "gripper_open":    gripper_open,
        }
