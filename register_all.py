"""
Run register_views on every acquisition folder inside a parent folder (e.g. a day's modulation data).

Subfolders without an MMStack .ome.tif are skipped, as are ones already registered in --out
(unless --force). A failure on one folder is reported and the rest still run.

Usage
-----
    conda run -n image python register_all.py <parent folder>
    conda run -n image python register_all.py <parent folder> --pattern "pockels_*" --reg <json> --out <dir>
"""
import argparse
from pathlib import Path

from register_views import DEFAULT_OUT, DEFAULT_REG, load_warp, register_one


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("parent", type=Path, help="folder containing one subfolder per acquisition")
    p.add_argument("--pattern", default="*", help='subfolder glob, e.g. "pockels_*" (default: all)')
    p.add_argument("--reg", type=Path, default=DEFAULT_REG, help="bead_registration.json")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--force", action="store_true", help="re-register folders already in --out")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    warp, split = load_warp(args.reg)
    folders = sorted(f for f in args.parent.glob(args.pattern) if f.is_dir())
    todo = [f for f in folders if any(f.glob("*MMStack*.ome.tif"))]
    skipped = sorted(set(folders) - set(todo))
    if skipped:
        print(f"skipping {len(skipped)} folders with no MMStack .ome.tif: " + ", ".join(f.name for f in skipped))
    print(f"registration {args.reg} (split {split}); {len(todo)} folders -> {args.out}")

    failed = []
    for n, folder in enumerate(todo, 1):
        if (args.out / f"{folder.name}_registered.tif").exists() and not args.force:
            print(f"[{n}/{len(todo)}] {folder.name}: already registered, skipping (--force to redo)")
            continue
        print(f"[{n}/{len(todo)}] ", end="", flush=True)
        try:
            register_one(folder, warp, split, args.out, args.reg)
        except Exception as e:  # keep going; report at the end
            print(f"{folder.name}: FAILED ({e})")
            failed.append(folder.name)
    if failed:
        print(f"\n{len(failed)} failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
