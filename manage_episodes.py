"""
Episode data management tool.

Usage:
    python manage_episodes.py                                       # list all groups
    python manage_episodes.py --group default                       # list episodes in group
    python manage_episodes.py --group default --delete 0 3 5        # delete specific episodes
    python manage_episodes.py --group default --delete 2-7          # delete a range
    python manage_episodes.py --group default --delete-all          # delete all episodes in group
    python manage_episodes.py --group default --delete-group        # delete group folder itself
    python manage_episodes.py --delete-all                          # delete all groups
    python manage_episodes.py --group src --move 0 3 5 --to dst     # move specific episodes to another group
    python manage_episodes.py --group src --move 2-7 --to dst       # move a range to another group
    python manage_episodes.py --group src --move-all --to dst       # move all episodes to another group
    python manage_episodes.py --migrate                             # migrate legacy episode_* dirs → 'default' group
"""

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config


# ── Load ──────────────────────────────────────────────────────────────────────

def _load_one(ep_dir: Path) -> dict:
    meta_path = ep_dir / "metadata.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    return {
        "dir":    ep_dir,
        "number": int(ep_dir.name.split("_")[1]),
        "label":  meta.get("label", "(no label)"),
        "frames": meta.get("frame_count", len(list(ep_dir.glob("frame_*.jpg")))),
    }


def load_episodes(group_dir: Path) -> list[dict]:
    ep_dirs = sorted(group_dir.glob("episode_*"))
    if not ep_dirs:
        return []
    with ThreadPoolExecutor(max_workers=min(16, len(ep_dirs))) as ex:
        results = list(ex.map(_load_one, ep_dirs))
    return sorted(results, key=lambda e: e["number"])


# ── Print ─────────────────────────────────────────────────────────────────────

def print_all_groups(data_dir: Path):
    groups = sorted(p.name for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if not groups:
        print("No groups found.")
        return
    print(f"\n{'group':<20}  {'episodes':>8}  {'frames':>8}")
    print("-" * 42)
    for g in groups:
        episodes = load_episodes(data_dir / g)
        print(f"{g:<20}  {len(episodes):>8}  {sum(ep['frames'] for ep in episodes):>8}")
    print()


def print_episodes(episodes: list[dict], group: str):
    if not episodes:
        print(f"No episodes in group '{group}'.")
        return
    print(f"\n[{group}]")
    print(f"{'#':>4}  {'frames':>7}  label")
    print("-" * 50)
    for ep in episodes:
        print(f"{ep['number']:>4}  {ep['frames']:>7}  {ep['label']}")
    print("-" * 50)
    print(f"{len(episodes)} episodes, {sum(ep['frames'] for ep in episodes)} frames total\n")


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

def delete_episodes(episodes: list[dict], targets: set[int], group_dir: Path):
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

    with ThreadPoolExecutor(max_workers=min(8, len(to_delete))) as ex:
        futures = {ex.submit(shutil.rmtree, ep["dir"]): ep for ep in to_delete}
        for f in as_completed(futures):
            print(f"  Deleted: {futures[f]['dir'].name}")

    _renumber(to_keep, group_dir)
    print(f"\nDone. {len(to_keep)} episode(s) remaining.")


def _renumber(episodes: list[dict], group_dir: Path):
    if not episodes:
        return

    # Phase 1: → temp names (parallel, 이름 충돌 방지)
    def to_tmp(ep):
        tmp = ep["dir"].with_name("_tmp_" + ep["dir"].name)
        ep["dir"].rename(tmp)
        return tmp

    with ThreadPoolExecutor(max_workers=min(16, len(episodes))) as ex:
        tmp_dirs = list(ex.map(to_tmp, episodes))

    # Phase 2: temp → 최종 이름 + metadata id 업데이트 (parallel)
    def to_final(args):
        new_id, tmp = args
        final = group_dir / f"episode_{new_id:04d}"
        tmp.rename(final)
        meta_path = final / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta["id"] = new_id
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    with ThreadPoolExecutor(max_workers=min(16, len(tmp_dirs))) as ex:
        list(ex.map(to_final, enumerate(tmp_dirs)))

    print(f"  Renumbered: episode_0000 ~ episode_{len(tmp_dirs)-1:04d}")


# ── Move ─────────────────────────────────────────────────────────────────────

def move_episodes(episodes: list[dict], targets: set[int], src_dir: Path, dst_dir: Path):
    to_move = [ep for ep in episodes if ep["number"] in targets]
    to_keep = [ep for ep in episodes if ep["number"] not in targets]

    if not to_move:
        print("No matching episodes to move.")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_offset = len(list(dst_dir.glob("episode_*")))

    print(f"\nMoving {len(to_move)} episode(s) → '{dst_dir.name}':")
    for ep in to_move:
        print(f"  episode_{ep['number']:04d}  ({ep['frames']} frames)  \"{ep['label']}\"")

    ans = input("\nMove? [y/N] ").strip().lower()
    if ans != "y":
        print("Cancelled.")
        return

    # 대상 그룹에 이름 충돌 방지용 임시 이름으로 이동 (parallel)
    def move_one(args):
        idx, ep = args
        tmp = dst_dir / f"_mv_{idx:04d}_{ep['dir'].name}"
        shutil.move(str(ep["dir"]), str(tmp))
        return tmp

    with ThreadPoolExecutor(max_workers=min(8, len(to_move))) as ex:
        tmp_dirs = list(ex.map(move_one, enumerate(to_move)))

    # 대상 그룹에서 최종 번호로 rename + metadata 업데이트 (parallel)
    def finalize(args):
        new_id, tmp = args
        final = dst_dir / f"episode_{new_id:04d}"
        tmp.rename(final)
        meta_path = final / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta["id"] = new_id
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    with ThreadPoolExecutor(max_workers=min(16, len(tmp_dirs))) as ex:
        list(ex.map(finalize, enumerate(tmp_dirs, start=dst_offset)))

    print(f"  → '{dst_dir.name}': episode_{dst_offset:04d} ~ episode_{dst_offset + len(to_move) - 1:04d}")

    _renumber(to_keep, src_dir)
    print(f"\nDone. '{src_dir.name}' has {len(to_keep)} episode(s) remaining.")


# ── Migrate ───────────────────────────────────────────────────────────────────

def migrate(data_dir: Path):
    old_eps = sorted(data_dir.glob("episode_*"))
    if not old_eps:
        print("No legacy episodes found.")
        return

    default_dir = data_dir / config.DEFAULT_GROUP
    default_dir.mkdir(exist_ok=True)

    existing = list(default_dir.glob("episode_*"))
    if existing:
        print(f"Warning: '{config.DEFAULT_GROUP}' already has {len(existing)} episodes. 전부 renumber됩니다.")
        ans = input("Continue? [y/N] ").strip().lower()
        if ans != "y":
            print("Cancelled.")
            return

    print(f"\nMigrating {len(old_eps)} episodes → group '{config.DEFAULT_GROUP}'...")

    def move_one(ep_dir):
        dest = default_dir / ep_dir.name
        if dest.exists():
            # 이름 충돌 시 임시 이름으로 이동
            tmp = default_dir / ("_mig_" + ep_dir.name)
            shutil.move(str(ep_dir), str(tmp))
            return tmp.name
        shutil.move(str(ep_dir), str(dest))
        return dest.name

    with ThreadPoolExecutor(max_workers=min(8, len(old_eps))) as ex:
        futures = {ex.submit(move_one, d): d for d in old_eps}
        for f in as_completed(futures):
            print(f"  Moved: {futures[f].name}")

    episodes = load_episodes(default_dir)
    _renumber(episodes, default_dir)
    print(f"\nMigration complete. {len(episodes)} episodes in group '{config.DEFAULT_GROUP}'.")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",     default=config.DATA_DIR)
    ap.add_argument("--group",        metavar="NAME", help="group to operate on")
    ap.add_argument("--delete",       nargs="+", metavar="N", help="episode numbers or ranges (e.g. 0 3 5-8)")
    ap.add_argument("--delete-all",   action="store_true",   help="delete all episodes (--group 지정 시 해당 그룹만, 아니면 전체)")
    ap.add_argument("--delete-group", action="store_true",   help="그룹 폴더 자체를 삭제 (--group 필요)")
    ap.add_argument("--move",         nargs="+", metavar="N", help="episode numbers or ranges을 --to 그룹으로 이동")
    ap.add_argument("--move-all",     action="store_true",   help="그룹 전체를 --to 그룹으로 이동")
    ap.add_argument("--to",           metavar="GROUP",       help="이동 대상 그룹 (--move / --move-all 에서 사용)")
    ap.add_argument("--migrate",      action="store_true",   help=f"legacy episode_* dirs → '{config.DEFAULT_GROUP}' group으로 migration")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    if args.migrate:
        migrate(data_dir)
        return

    if args.group:
        group_dir = data_dir / args.group

        if args.delete_group:
            if not group_dir.exists():
                print(f"Group '{args.group}' not found.")
                return
            episodes = load_episodes(group_dir)
            print(f"\nGroup '{args.group}': {len(episodes)} episodes")
            ans = input(f"Delete group '{args.group}' and all its contents? [y/N] ").strip().lower()
            if ans == "y":
                shutil.rmtree(group_dir)
                print(f"Deleted group: {args.group}")
            return

        group_dir.mkdir(exist_ok=True)
        episodes = load_episodes(group_dir)
        print_episodes(episodes, args.group)

        if args.move or args.move_all:
            if not args.to:
                print("--to GROUP 을 지정해줘.")
                return
            dst_dir = data_dir / args.to
            targets = {ep["number"] for ep in episodes} if args.move_all else parse_targets(args.move)
            move_episodes(episodes, targets, group_dir, dst_dir)
        elif args.delete_all:
            targets = {ep["number"] for ep in episodes}
            delete_episodes(episodes, targets, group_dir)
        elif args.delete:
            targets = parse_targets(args.delete)
            delete_episodes(episodes, targets, group_dir)
    else:
        print_all_groups(data_dir)

        if args.delete_all:
            ans = input("Delete ALL episodes in ALL groups? [y/N] ").strip().lower()
            if ans == "y":
                groups = sorted(p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))
                for g in groups:
                    shutil.rmtree(g)
                    print(f"  Deleted group: {g.name}")
                print("All groups deleted.")


if __name__ == "__main__":
    main()
