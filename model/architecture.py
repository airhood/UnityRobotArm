"""
Robot arm control model (behavior cloning) — CLIP-based.

Architecture:
  CLIP ViT-B/32 visual encoder  (frozen)  →  (512)  ─┐
                                                        │  element-wise product  →  (512)
  CLIP text encoder             (frozen)  →  (512)  ─┘
                                                        │
  State MLP: joint angles (5) + EE pos (3)  →  (64)  ─┘  concat  →  (576)
                                                                          │
                                                                   Action MLP  →  (256)
                                                                          │
                                             ┌────────────────────────────┤
                                        Δpos (3)                    gripper (1)

Why CLIP:
  - Visual and text encoders share the same embedding space (already aligned).
  - "move up" and an image of the arm moving up are already close in CLIP space.
  - Only the Action MLP and State MLP need to be learned → few parameters,
    stable training even with a modest dataset.
"""

import torch
import torch.nn as nn


class RobotArmModel(nn.Module):
    def __init__(
        self,
        clip_dim: int = 512,      # CLIP ViT-B/32 output dim
        joint_dim: int = 5,
        hidden_dim: int = 256,
        action_dim: int = 3,
    ):
        super().__init__()

        # State encoder: joint angles (5) + EE position (3) → 64-dim
        self.state_encoder = nn.Sequential(
            nn.Linear(joint_dim + 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Action MLP: fused vision-language (512) + state (64) → hidden
        self.action_mlp = nn.Sequential(
            nn.Linear(clip_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.position_head = nn.Linear(hidden_dim, action_dim)
        self.gripper_head  = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        image_feat: torch.Tensor,   # (B, 512)  — pre-computed CLIP visual feat
        text_feat:  torch.Tensor,   # (B, 512)  — pre-computed CLIP text feat
        joint_angles: torch.Tensor, # (B, 5)    degrees
        ee_position:  torch.Tensor, # (B, 3)    Unity world coords
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fuse vision + language via element-wise product (both in CLIP space)
        vl_feat = image_feat * text_feat                                # (B, 512)

        state_feat = self.state_encoder(
            torch.cat([joint_angles, ee_position], dim=-1)
        )                                                               # (B, 64)

        h = self.action_mlp(torch.cat([vl_feat, state_feat], dim=-1))  # (B, hidden)
        return self.position_head(h), self.gripper_head(h)
