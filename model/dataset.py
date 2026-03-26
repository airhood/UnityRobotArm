import json
from pathlib import Path
from typing import Callable, Literal

import clip
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class RobotArmDataset(Dataset):
    """
    Behavior-cloning dataset built from DataCollector episodes.

    Each sample (t → t+1):
        Input:  image_t (CLIP-preprocessed), text_tokens, joint_angles_t, ee_position_t
        Target: ee_position_{t+1}

    Pass clip_preprocess from clip.load() so images are prepared correctly for CLIP.
    If not provided, falls back to standard ImageNet normalisation.
    """

    def __init__(
        self,
        data_dir: str = "data/episodes",
        image_size: int = 224,
        split: Literal["train", "val"] = "train",
        split_ratio: float = 0.9,
        clip_preprocess: Callable | None = None,
    ):
        self.transform = clip_preprocess or T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.samples: list[dict] = []
        self._load(Path(data_dir), split, split_ratio)

    def _load(self, root: Path, split: str, ratio: float):
        episodes = sorted(root.glob("episode_*"))
        cut = int(len(episodes) * ratio)
        episodes = episodes[:cut] if split == "train" else episodes[cut:]

        for ep_dir in episodes:
            meta_p = ep_dir / "metadata.json"
            states_p = ep_dir / "states.json"
            if not meta_p.exists() or not states_p.exists():
                continue

            with open(meta_p) as f:
                label = json.load(f).get("label", "")
            with open(states_p) as f:
                states = json.load(f)

            for i in range(len(states) - 1):
                s, sn = states[i], states[i + 1]
                if not Path(s["image_path"]).exists():
                    continue
                self.samples.append({
                    "image_path": s["image_path"],
                    "text": label,
                    "joint_angles": s["joint_angles"],
                    "ee_position": s["ee_position"],
                    "target_position": sn["ee_position"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = self.transform(Image.open(s["image_path"]).convert("RGB"))
        return {
            "image": image,
            "text_tokens": clip.tokenize([s["text"]], truncate=True).squeeze(0),  # (77,)
            "joint_angles": torch.tensor(s["joint_angles"], dtype=torch.float32),
            "ee_position": torch.tensor(s["ee_position"], dtype=torch.float32),
            "target_position": torch.tensor(s["target_position"], dtype=torch.float32),
        }
