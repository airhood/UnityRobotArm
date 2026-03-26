"""
Behavior-cloning training for the CLIP-based robot arm model.

Usage:
    python -m model.train                          # train from scratch
    python -m model.train --resume                 # resume from best_model.pt
    python -m model.train --resume path/to/ckpt.pt # resume from specific checkpoint
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import clip  # pip install git+https://github.com/openai/CLIP.git

import config
from model.dataset import RobotArmDataset
from model.architecture import RobotArmModel
from utils.log_setup import setup as setup_logging

logger = setup_logging("train")


def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", nargs="?", const=True, default=False,
                    metavar="CHECKPOINT",
                    help="resume from checkpoint (default: best_model.pt)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── CLIP (frozen) ──────────────────────────────────────────────────────
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    logger.info("CLIP ViT-B/32 loaded and frozen")

    # ── Dataset ────────────────────────────────────────────────────────────
    train_ds = RobotArmDataset(
        config.DATA_DIR, config.IMAGE_SIZE, "train",
        clip_preprocess=clip_preprocess,
    )
    val_ds = RobotArmDataset(
        config.DATA_DIR, config.IMAGE_SIZE, "val",
        clip_preprocess=clip_preprocess,
    )
    logger.info(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    if len(train_ds) == 0:
        logger.error("No training data. Run collect_data.py first.")
        return

    train_dl = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── Model (only Action MLP + State MLP are trained) ────────────────────
    model = RobotArmModel(
        clip_dim=512,
        hidden_dim=config.HIDDEN_DIM,
        action_dim=config.ACTION_DIM,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    loss_fn = nn.MSELoss()

    ckpt_dir = Path(config.MODEL_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    start_epoch = 1

    # ── Resume ─────────────────────────────────────────────────────────────
    if args.resume:
        ckpt_path = Path(args.resume) if isinstance(args.resume, str) else ckpt_dir / "best_model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val = ckpt.get("val_loss", float("inf"))
            logger.info(f"Resumed from {ckpt_path} (epoch {start_epoch - 1}, val_loss={best_val:.6f})")
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path} — starting from scratch")

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        total = 0.0
        for batch in train_dl:
            images      = batch["image"].to(device)        # already CLIP-preprocessed
            tokens      = batch["text_tokens"].to(device)  # clip.tokenize output
            ja          = batch["joint_angles"].to(device)
            ee          = batch["ee_position"].to(device)
            target_pos  = batch["target_position"].to(device)

            with torch.no_grad():
                img_feat  = clip_model.encode_image(images).float()   # (B, 512)
                text_feat = clip_model.encode_text(tokens).float()    # (B, 512)
                # L2-normalise so element-wise product is a cosine-style similarity
                img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            delta_pred, _ = model(img_feat, text_feat, ja, ee)
            loss = loss_fn(delta_pred, target_pos - ee)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        train_loss = total / len(train_dl)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in val_dl:
                images     = batch["image"].to(device)
                tokens     = batch["text_tokens"].to(device)
                ja         = batch["joint_angles"].to(device)
                ee         = batch["ee_position"].to(device)
                target_pos = batch["target_position"].to(device)

                img_feat  = clip_model.encode_image(images).float()
                text_feat = clip_model.encode_text(tokens).float()
                img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

                delta_pred, _ = model(img_feat, text_feat, ja, ee)
                val_total += loss_fn(delta_pred, target_pos - ee).item()

        val_loss = val_total / max(len(val_dl), 1)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
            f"train={train_loss:.6f} | val={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                },
                ckpt_dir / "best_model.pt",
            )

        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                ckpt_dir / f"epoch_{epoch:04d}.pt",
            )

    logger.info(f"Training complete. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    train()
