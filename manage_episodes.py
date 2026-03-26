"""
Episode data management tool.

Usage:
    python manage_episodes.py                         # list episodes
    python manage_episodes.py --delete 0 3 5          # delete specific episodes
    python manage_episodes.py --delete 2-7            # delete a range
    python manage_episodes.py --delete-all            # delete all episodes
    python manage_episodes.py --data-dir data/episodes
"""

import argparse
import json
import shutil
from pathlib import Path

import config


# ── Load episodes ────────────────────────────────────────────────────────────

def load_episodes(data_dir: Path) -> list[dict]:
    episodes = []
    for ep_dir in sorted(data_dir.glob("episode_*")):
        meta_path = ep_dir / "metadata.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        episodes.append({
            "dir":    ep_dir,
            "number": int(ep_dir.name.split("_")[1]),
            "label":  meta.get("label", "(no label)"),
            "frames": meta.get("frame_count", len(list(ep_dir.glob("frame_*.jpg")))),
        })
    return episodes


# ── Print ─────────────────────────────────────────────────────────────────────

def print_episodes(episodes: list[dict]):
    if not episodes:
        print("No episodes found.")
        return
    print(f"\n{'#':>4}  {'frames':>7}  label")
    print("-" * 50)
    for ep in episodes:
        print(f"{ep['number']:>4}  {ep['frames']:>7}  {ep['label']}")
    total_frames = sum(ep["frames"] for ep in episodes)
    print("-" * 50)
    print(f"{len(episodes)} episodes, {total_frames} frames total\n")


# ── Parse range ("2-7" → {2,3,4,5,6,7}) ─────────────────────────────────────

def parse_targets(tokens: list[str]) -> set[int]:
    result = set()
    for t in tokens:
        if "-" in t:
            a, b = t.split("-", 1)
            result.update(range(int(a), int(b) + 1))
        else:
            result.add(int(t))
    return result


# ── Delete + renumber ─────────────────────────────────────────────────────────

def delete_episodes(episodes: list[dict], targets: set[int], data_dir: Path):
    to_delete = [ep for ep in episodes if ep["number"] in targets]
    to_keep   = [ep for ep in episodes if ep["number"] not in targets]

    if not to_delete:
        print("No matching episodes to delete.")
        return

    print("\nTo be deleted:")
    for ep in to_delete:
        print(f"  episode_{ep['number']:04d}  ({ep['frames']} frames)  \"{ep['label']}\"")

    ans = input("\nDelete? [y/N] ").strip().lower()
    if ans != "y":
        print("Cancelled.")
        return

    for ep in to_delete:
        shutil.rmtree(ep["dir"])
        print(f"  Deleted: {ep['dir'].name}")

    _renumber(to_keep, data_dir)

    print(f"\nDone. {len(to_keep)} episode(s) remaining.")


def _renumber(episodes: list[dict], data_dir: Path):
    """Renumber remaining episodes from 0 (to prevent ID collisions)."""
    # Move to temp names first to avoid collisions mid-rename
    tmp_dirs = []
    for ep in episodes:
        tmp = ep["dir"].with_name("_tmp_" + ep["dir"].name)
        ep["dir"].rename(tmp)
        tmp_dirs.append(tmp)

    for new_id, tmp in enumerate(tmp_dirs):
        final = data_dir / f"episode_{new_id:04d}"
        tmp.rename(final)

        meta_path = final / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta["id"] = new_id
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    if tmp_dirs:
        print(f"  Renumbered: episode_0000 ~ episode_{len(tmp_dirs)-1:04d}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",   default=config.DATA_DIR)
    ap.add_argument("--delete",     nargs="+", metavar="N", help="episode numbers or ranges to delete (e.g. 0 3 5-8)")
    ap.add_argument("--delete-all", action="store_true",    help="delete all episodes")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    episodes = load_episodes(data_dir)
    print_episodes(episodes)

    if args.delete_all:
        targets = {ep["number"] for ep in episodes}
        delete_episodes(episodes, targets, data_dir)
    elif args.delete:
        targets = parse_targets(args.delete)
        delete_episodes(episodes, targets, data_dir)


if __name__ == "__main__":
    main()
