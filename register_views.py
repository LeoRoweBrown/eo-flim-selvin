"""
Register the two camera views onto one image with two channels, using bead_registration.json.

Each frame of the raw stack holds both views, split at the row saved with the registration:
top = channel 0 (kept as is, defines the pixel grid), bottom = channel 1 (warped onto the top
grid with the bead affine, bilinear interpolation). Pixels the bottom view does not cover are NaN.

Output: <name>_registered.tif, an ImageJ hyperstack (T, C, Y, X) float32 with the raw counts
(no background subtraction), one per input.

Usage
-----
    conda run -n image python register_views.py <acquisition folder or .tif> [...]
    conda run -n image python register_views.py <folder> --reg <bead_registration.json> --out <dir>
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile

DEFAULT_REG = Path(r"E:\20261019\20261019_analysis\bead_registration"
                   r"\bead_stack_0.69vrms_lc_200ms_20mW_2\bead_registration.json")
DEFAULT_OUT = Path(r"E:\20261019\20261019_analysis")


def load_warp(path):
    """(2x3 warp, split) from bead_registration.json; also reads the older multi-model format."""
    j = json.loads(Path(path).read_text())
    if "warp" in j:
        w = j["warp"]
    elif "models" in j:
        w = j["models"]["affine"]["params"][:2]
    else:
        raise SystemExit(f"{path}: no 'warp' or 'models.affine' entry")
    if j.get("split") is None:
        raise SystemExit(f"{path}: no 'split' entry")
    return np.array(w, np.float64), int(j["split"])


def find_stack(path):
    """The .tif itself, or the first .tif in a folder."""
    if path.is_dir():
        files = sorted(path.glob("*.tif"))
        if not files:
            raise FileNotFoundError(f"no tif in {path}")
        return files[0]
    return path


def register(raw, warp, split):
    """(T, 2, split, W) float32: top view and the bottom view warped onto it."""
    W = raw.shape[-1]
    out = np.empty((len(raw), 2, split, W), np.float32)
    for t, f in enumerate(raw):
        f = f.astype(np.float32)
        bot = f[split:2 * split]
        out[t, 0] = f[:split]
        out[t, 1] = cv2.warpAffine(bot, warp, (W, split), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    return out


def register_one(path, warp, split, out, reg_path):
    """Register one acquisition folder or .tif and save <name>_registered.tif in out."""
    file = find_stack(path)
    name = path.name if path.is_dir() else file.name.removesuffix(".tif")
    raw = tifffile.imread(file)
    raw = raw.reshape(-1, *raw.shape[-2:])
    if raw.shape[-2] < split + 1:
        raise ValueError(f"{file}: {raw.shape[-2]} rows, too few for split {split}")
    reg = register(raw, warp, split)
    dst = out / f"{name}_registered.tif"
    tifffile.imwrite(dst, reg, imagej=True, metadata=dict(
        axes="TCYX", Info=f"source={file}\nregistration={reg_path}\nsplit={split}\n"
                          "C0=top view, C1=bottom view warped onto top"))
    cover = np.isfinite(reg[0, 1]).mean()
    print(f"{name}: {len(raw)} frames -> {dst.name} {reg.shape}, bottom view covers {cover:.0%} of the grid")
    return dst


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, nargs="+", help="acquisition folder(s) or .tif file(s)")
    p.add_argument("--reg", type=Path, default=DEFAULT_REG, help="bead_registration.json")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--files-in-root", action="store_true",
                   help="look for .tif files in subfolders of each data folder")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("Starting registration of {} acquisition(s) in {}".format(len(args.data), args.data[0]))

    warp, split = load_warp(args.reg)
    print(f"registration json file: {args.reg} (split {split})")

    # get list of subfolders that contain .tif files if --files-in-root is specified
    if not args.files_in_root:
        subfolders = []
        for folder in args.data:
            if folder.is_dir():
                subfolders.extend(sorted(f for f in folder.glob("*") if f.is_dir() and any(f.glob("*.tif"))))
            else:
                print(f"{folder} is not a directory, skipping")
        args.data = subfolders
        print(f"Found {len(args.data)} subfolders with .tif files")
    
    for path in args.data:
        register_one(path, warp, split, args.out, args.reg)


if __name__ == "__main__":
    main()
