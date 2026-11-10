"""
Modulation depth from registered two-channel phase scans (output of register_views / register_all).

Each <name>_registered.tif is (T, C, Y, X): frames = RF phase steps (default 0, 30, ..., 150 deg),
C0 = top view (ungated, U), C1 = bottom view warped onto it (gated, G).

Per dataset
-----------
1. Subtract the camera offset (median of the unilluminated left/right margins of U, per frame).
2. Frame normalisation (default): U + G collects all the emission, so it should not depend on
   phase; each pixel is rescaled so its U + G equals the phase-mean. This removes excitation
   fluctuations between frames.
3. Bin, then per pixel least-squares Fourier fit in phase,
       I(phi) = a0 + sum_n [a_n cos(n k phi) + b_n sin(n k phi)],  k = 2 pi / period,
   and modulation depth m1 = sqrt(a1^2 + b1^2) / a0 (= (max-min)/(max+min) of the fundamental).
4. Mask: pixels where both U and G are covered and above --mask-thr x their 99th percentile.
5. Central ROI (centroid of the mask, or --roi-center): summed U and G against phase with fits.

Outputs (in --out): <name>.png (ROI intensity vs phase, m1 maps of U, G and their average)
and summary.csv (ROI and map-median m1 per dataset).

Usage
-----
    conda run -n image python modulation.py
    conda run -n image python modulation.py <registered folder or *_registered.tif ...> --out <dir>
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

import matplotlib.pyplot as plt

DEFAULT_DATA = Path(r"E:\20261019\20261019_analysis\registered\modulation_data")
DEFAULT_OUT = Path(r"E:\20261019\20261019_analysis\modulation_results")


# ------------------------------------------------------------------- fitting
def design(phases, period, nh):
    k = 2 * np.pi / period
    cols = [np.ones_like(phases)]
    for n in range(1, nh + 1):
        cols += [np.cos(n * k * phases), np.sin(n * k * phases)]
    return np.stack(cols, 1)


def fit_harmonics(stack, phases, period, nh):
    """Least-squares Fourier coefficients [a0, a1, b1, a2, b2, ...] along axis 0.

    Uses the pseudo-inverse rather than lstsq so NaN (uncovered) pixels stay NaN on their own."""
    D = design(phases, period, nh)
    coef = np.linalg.pinv(D) @ stack.reshape(len(phases), -1)
    return coef.reshape((D.shape[1],) + stack.shape[1:])


def m1(coef):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.hypot(coef[1], coef[2]) / coef[0]


# -------------------------------------------------------------------- images
def background(U, margin):
    """Per-frame camera offset from the unilluminated left/right margins of the top view."""
    edge = np.concatenate([U[:, :, :margin], U[:, :, -margin:]], axis=2)
    return np.median(edge, axis=(1, 2))


def normalize_frames(U, G):
    """Rescale every pixel so U + G equals its phase-mean; returns U, G and the per-frame
    total U + G relative to its median (how much the excitation fluctuated)."""
    T = U + G
    tot = np.nansum(T, axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(T > 0, T.mean(0) / T, 1.0)
    return U * s, G * s, tot / np.median(tot)


def bin_stack(a, b):
    """reshape into 4D to bin the image by averaging"""
    h, w = a.shape[-2] // b * b, a.shape[-1] // b * b
    a = a[..., :h, :w]
    return a.reshape(a.shape[:-2] + (h // b, b, w // b, b)).mean(axis=(-3, -1))


# ---------------------------------------------------------------------- main
def process(path, args):
    reg = tifffile.imread(path).astype(np.float32)
    if reg.ndim == 3:                                   # a single frame saved as (C, Y, X)
        reg = reg[None]
    U, G = reg[:, 0], reg[:, 1]
    bg = background(U, args.bg_margin) if args.bg is None else np.full(len(U), args.bg)
    U, G = U - bg[:, None, None], G - bg[:, None, None]
    if args.frame_norm:
        U, G, tot = normalize_frames(U, G)
    else:
        tot = np.nansum(U + G, axis=(1, 2))
        tot = tot / np.median(tot)

    phases = np.arange(len(U)) * args.phase_step
    nh = min(args.harmonics, (len(phases) - 1) // 2)
    Ub, Gb = bin_stack(U, args.bin), bin_stack(G, args.bin)   # bins touching NaN (uncovered) stay NaN
    cU, cG = fit_harmonics(Ub, phases, args.period, nh), fit_harmonics(Gb, phases, args.period, nh)
    mU, mG = m1(cU), m1(cG)
    mAvg = 0.5 * (mU + mG)

    U_dc, G_dc = cU[0], cG[0]
    mask = np.isfinite(U_dc) & np.isfinite(G_dc)
    mask &= (U_dc > args.mask_thr * np.nanpercentile(U_dc, 99)) & (G_dc > args.mask_thr * np.nanpercentile(G_dc, 99))
    mask = ndi.binary_erosion(mask, iterations=1)
    if not mask.any():
        raise ValueError("empty mask: no pixels with signal in both channels")

    # central ROI, full-resolution pixels
    if args.roi_center is not None:
        cy, cx = args.roi_center
    else:
        cy, cx = ((np.array(ndi.center_of_mass(mask)) + 0.5) * args.bin).round().astype(int)
    h = args.roi // 2
    box = (slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h))
    u, g = U[:, box[0], box[1]].sum((1, 2)), G[:, box[0], box[1]].sum((1, 2))
    if not (np.isfinite(u).all() and np.isfinite(g).all()):
        raise ValueError(f"ROI at (y={cy}, x={cx}) reaches outside the bottom view's footprint")
    ru, rg = fit_harmonics(u[:, None], phases, args.period, nh)[:, 0], fit_harmonics(g[:, None], phases, args.period, nh)[:, 0]

    row = dict(name=path.name.removesuffix("_registered.tif"), n_phases=len(phases),
               roi_y=cy, roi_x=cx, roi_U_m1=float(m1(ru)), roi_G_m1=float(m1(rg)),
               roi_avg_m1=float(0.5 * (m1(ru) + m1(rg))),
               median_U_m1=float(np.nanmedian(mU[mask])), median_G_m1=float(np.nanmedian(mG[mask])),
               median_avg_m1=float(np.nanmedian(mAvg[mask])),
               p99_avg_m1=float(np.nanpercentile(mAvg[mask], 99)),
               frame_total_min=float(tot.min()), frame_total_max=float(tot.max()), bg=float(bg.mean()))
    plot(row, phases, u, g, ru, rg, nh, (mU, mG, mAvg), mask, box, args)
    return row


def plot(row, phases, u, g, ru, rg, nh, maps, mask, box, args):
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    a = ax[0, 0]
    dense = np.linspace(0, args.period, 361)
    for y, c, label, col in ((u, ru, "U (top)", "C0"), (g, rg, "G (bottom)", "C3")):
        a.plot(phases, y / c[0], "o", c=col, label=f"{label}: m1 = {m1(c):.3f}")
        a.plot(dense, design(dense, args.period, nh) @ c / c[0], "-", c=col, lw=1)
    a.set_xlabel("RF phase [deg]")
    a.set_ylabel("ROI intensity / mean")
    a.set_title(f"central {args.roi} px ROI at (y={row['roi_y']}, x={row['roi_x']})"
                + ("" if args.frame_norm else ", no frame normalisation"))
    a.legend()

    vmax = args.vmax or np.ceil(np.nanpercentile(maps[2][mask], 99) * 20) / 20
    titles = ("U modulation depth m1", "G modulation depth m1", "average (U + G) / 2 modulation depth m1")
    y0, x0 = box[0].start / args.bin, box[1].start / args.bin
    for a, m, t in zip((ax[0, 1], ax[1, 0], ax[1, 1]), maps, titles):
        im = a.imshow(np.where(mask, m, np.nan), cmap="viridis", vmin=0, vmax=vmax)
        a.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), args.roi / args.bin, args.roi / args.bin,
                                  fill=False, ec="w", lw=1))
        a.set_title(f"{t} (median {np.nanmedian(m[mask]):.3f})")
        a.set_xlabel(f"x [{args.bin} px bins]")
        a.set_ylabel(f"y [{args.bin} px bins]")
        fig.colorbar(im, ax=a, fraction=0.03)
    fig.suptitle(row["name"])
    fig.tight_layout()
    fig.savefig(args.out / f"{row['name']}.png", dpi=90)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, nargs="*", default=[DEFAULT_DATA],
                   help="folder of *_registered.tif and/or individual files")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--phase-step", type=float, default=30.0, help="RF phase step between frames [deg]")
    p.add_argument("--period", type=float, default=180.0, help="gating period in RF phase [deg]")
    p.add_argument("--harmonics", type=int, default=2, help="max harmonics in the fit (capped by #phases)")
    p.add_argument("--bin", type=int, default=8, help="spatial binning for the maps")
    p.add_argument("--roi", type=int, default=200, help="ROI side length [full-res px]")
    p.add_argument("--roi-center", type=int, nargs=2, default=None, metavar=("Y", "X"),
                   help="ROI centre [full-res px] (default: centroid of the mask)")
    p.add_argument("--mask-thr", type=float, default=0.3, help="valid if U and G mean > thr * 99th pct")
    p.add_argument("--no-frame-norm", dest="frame_norm", action="store_false",
                   help="skip the per-pixel U + G frame normalisation")
    p.add_argument("--bg", type=float, default=None, help="fixed camera offset (default: from image margins)")
    p.add_argument("--bg-margin", type=int, default=250, help="margin width [px] used for the offset")
    p.add_argument("--vmax", type=float, default=None, help="colour-scale max for the m1 maps")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = []
    for d in args.data:
        files += sorted(d.glob("*_registered.tif")) if d.is_dir() else [d]
    if not files:
        raise SystemExit(f"no *_registered.tif in {', '.join(map(str, args.data))}")

    rows = []
    for n, f in enumerate(files, 1):
        try:
            row = process(f, args)
        except Exception as e:  # keep going; report at the end
            print(f"[{n}/{len(files)}] {f.name}: FAILED ({e})")
            continue
        rows.append(row)
        print(f"[{n}/{len(files)}] {row['name']}: ROI m1 U {row['roi_U_m1']:.3f}  G {row['roi_G_m1']:.3f}  "
              f"avg {row['roi_avg_m1']:.3f}   map median avg {row['median_avg_m1']:.3f}   "
              f"U+G range {row['frame_total_min']:.2f}-{row['frame_total_max']:.2f}")
    if rows:
        pd.DataFrame(rows).to_csv(args.out / "summary.csv", index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    matplotlib.use("Agg")  # batch only: not on import, so live_modulation keeps an interactive backend
    main()
