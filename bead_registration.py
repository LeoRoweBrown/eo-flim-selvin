"""
Two-view registration from beads: an affine transform mapping the top view onto the bottom view.

Steps
-----
1. Split row: middle of the dark gap between the views (or --split).
2. Pick best focus separately for each view (bright DoG peaks per slice) and average a few
   slices around it. Single frames and time series work too.
3. Detect beads: difference-of-Gaussians local maxima above k * shot-noise model, sub-pixel
   positions from a 2D Gaussian fit.
4. Match beads top -> bottom with no prior registration: the translation is voted over all
   bead-pair offsets, then an affine is refined on mutual nearest neighbours with a shrinking
   match radius, and RANSAC drops wrong matches.
5. Quality: 5-fold cross-validated residual at the beads (mostly bead localisation noise) and
   the bootstrap error of the transform itself over the field (what a mapped point inherits).

The transform maps top-view (x, y) -> bottom-view (x, y) in bottom-local coordinates
(y measured from the split row). "warp" in the output is the 2x3 matrix for
cv2.warpAffine(bottom_view, warp, (W, split), flags=WARP_INVERSE_MAP), which resamples
the bottom view onto the top view's pixel grid.

Usage
-----
    conda run -n image python bead_registration.py
    conda run -n image python bead_registration.py --stack <folder or file.ome.tif> --out <dir>
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from skimage.measure import ransac
from skimage.transform import AffineTransform

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_STACK = Path(r"D:\Leo\20261019\20261019\bead_stack_0.69vrms_lc_200ms_20mW_2")
DEFAULT_OUT = Path(r"D:\Leo\20261019\20261019_analysis\bead_registration")


# ------------------------------------------------------------------- images
def load_stack(path):
    """(n, Y, X) float32 from an .ome.tif, or the first MMStack .ome.tif in a folder."""
    if path.is_dir():
        files = sorted(path.glob("*MMStack*.ome.tif"))
        if not files:
            raise FileNotFoundError(f"no MMStack .ome.tif in {path}")
        path = files[0]
    a = tifffile.imread(path)
    return a.reshape(-1, *a.shape[-2:]).astype(np.float32), path


def detect_split(stack, lo=0.3, hi=0.7):
    """Row in the middle of the dark gap between the two views (searched in lo..hi of the height)."""
    prof = ndi.gaussian_filter1d(np.median(stack.sum(0), axis=1), 5)
    a, b = int(lo * len(prof)), int(hi * len(prof))
    seg = prof[a:b]
    dark = seg < seg.min() + 0.1 * (np.percentile(seg, 90) - seg.min())
    lab, n = ndi.label(dark)
    sizes = ndi.sum(dark, lab, range(1, n + 1))
    rows = np.nonzero(lab == 1 + int(np.argmax(sizes)))[0]
    return a + int(round(rows.mean()))


def focus_image(stack, rows, nz, sigma=1.5, n_peaks=200):
    """Mean of nz slices around the best-focus slice of stack[:, rows].

    Focus metric: mean of the n_peaks brightest DoG pixels, so that bead peaks rather
    than noise drive it (needed for the faint gated view)."""
    energy = []
    for s in stack[:, rows]:
        dog = ndi.gaussian_filter(s, sigma) - ndi.gaussian_filter(s, 4 * sigma)
        energy.append(np.partition(dog.ravel(), -n_peaks)[-n_peaks:].mean())
    energy = np.array(energy)
    z = int(np.argmax(energy))
    lo, hi = max(z - nz // 2, 0), min(z + nz // 2 + 1, len(stack))
    return stack[lo:hi, rows].mean(0), z, energy


def detect(img, sigma, k, min_sep, border=8):
    """Bead candidates: DoG local maxima above k * local noise.

    Noise follows a shot-noise model, sigma_dog = c * sqrt(local signal + read^2), with c and the
    read term set robustly from the image, so the threshold adapts to the illumination."""
    dog = ndi.gaussian_filter(img, sigma) - ndi.gaussian_filter(img, 4 * sigma)
    offset = np.percentile(img, 5)
    level = np.clip(ndi.gaussian_filter(img, 25) - offset, 0, None)
    dark = level < np.percentile(level, 20)
    read = 1.4826 * np.median(np.abs(dog[dark] - np.median(dog[dark])))
    shape = np.sqrt(level + 1.0)
    c = 1.4826 * np.median(np.abs(dog / shape))
    noise = np.sqrt((c * shape) ** 2 + read ** 2)
    snr = dog / noise
    peaks = (dog == ndi.maximum_filter(dog, size=2 * min_sep + 1)) & (snr > k)
    peaks[:border], peaks[-border:], peaks[:, :border], peaks[:, -border:] = False, False, False, False
    y, x = np.nonzero(peaks)
    return np.column_stack([x, y]).astype(float), snr[y, x]


def gauss_refine(img, pts, half=4):
    """Sub-pixel (x, y) from a 2D Gaussian + offset fit; drops failed fits."""
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    out, keep = [], []
    for i, (x0, y0) in enumerate(pts.astype(int)):
        patch = img[y0 - half:y0 + half + 1, x0 - half:x0 + half + 1]
        if patch.shape != yy.shape:
            continue
        b, a = np.median(patch), patch.max() - np.median(patch)

        def res(p):
            return (p[0] + p[1] * np.exp(-((xx - p[2]) ** 2 + (yy - p[3]) ** 2) / (2 * p[4] ** 2)) - patch).ravel()

        r = least_squares(res, [b, a, 0, 0, 1.5], bounds=([-np.inf, 0, -half, -half, 0.5], [np.inf, np.inf, half, half, half]))
        if r.success and abs(r.x[2]) < 2 and abs(r.x[3]) < 2:
            out.append([x0 + r.x[2], y0 + r.x[3]])
            keep.append(i)
    return np.array(out), np.array(keep, int)


# ----------------------------------------------------------------- matching
def vote_translation(top, bot, bin_px):
    """Translation t maximising #pairs with bot ~ top + t, and its peak / next-peak ratio.

    bin_px must cover the smear from the unmodelled scale and rotation (~1% over the field)."""
    d = (bot[None, :, :] - top[:, None, :]).reshape(-1, 2)
    lim = np.abs(d).max() + bin_px
    edges = np.arange(-lim, lim + bin_px, bin_px)
    H, ex, ey = np.histogram2d(d[:, 0], d[:, 1], bins=[edges, edges])
    H = ndi.uniform_filter(H, 3)
    i, j = np.unravel_index(np.argmax(H), H.shape)
    c = np.array([ex[i] + bin_px / 2, ey[j] + bin_px / 2])
    near = np.linalg.norm(d - c, axis=1) < 2 * bin_px
    peak = H[i, j]
    w = 4  # exclude +-4 bins around the peak
    H[max(i - w, 0):i + w + 1, max(j - w, 0):j + w + 1] = 0
    return d[near].mean(0), peak / max(H.max(), 1e-9)


def mutual_nn(p, q, radius):
    """Index pairs (i, j) where p[i], q[j] are mutual nearest neighbours within radius."""
    D = np.linalg.norm(p[:, None] - q[None], axis=2)
    j = D.argmin(1)
    i_back = D.argmin(0)
    i = np.arange(len(p))
    ok = (i_back[j] == i) & (D[i, j] < radius)
    return i[ok], j[ok]


def fit_affine(src, dst):
    if hasattr(AffineTransform, "from_estimate"):  # skimage >= 0.26
        t = AffineTransform.from_estimate(src, dst)
        if not t:
            raise RuntimeError("affine fit failed")
        return t
    t = AffineTransform()
    if not t.estimate(src, dst):
        raise RuntimeError("affine fit failed")
    return t


def cv_rms(src, dst, folds=5, seed=0):
    idx = np.random.default_rng(seed).permutation(len(src))
    err = np.empty(len(src))
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        err[test] = np.linalg.norm(fit_affine(src[train], dst[train])(src[test]) - dst[test], axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


def linear_params(A):
    sx, sy = np.hypot(A[0, 0], A[1, 0]), np.hypot(A[0, 1], A[1, 1])
    rot = np.degrees(np.arctan2(A[1, 0], A[0, 0]))
    shear = np.degrees(np.arctan2(A[0, 1], A[1, 1])) + rot  # deviation of y-axis from perpendicular
    return dict(scale_x=float(sx), scale_y=float(sy), rot_deg=float(rot), shear_deg=float(shear))


# --------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stack", type=Path, default=DEFAULT_STACK, help="bead acquisition folder or .ome.tif")
    p.add_argument("--out", type=Path, default=None, help="output dir (default: <DEFAULT_OUT>/<stack folder name>)")
    p.add_argument("--split", type=int, default=None, help="first row of the bottom view (default: auto-detect)")
    p.add_argument("--nz", type=int, default=7, help="slices averaged around best focus")
    p.add_argument("--sigma", type=float, default=1.5, help="DoG inner sigma [px]")
    p.add_argument("--k", type=float, default=6.0, help="detection threshold [x noise]")
    p.add_argument("--radius", type=float, default=6.0, help="final match radius [px]")
    p.add_argument("--vote-bin", type=float, default=8.0, help="translation vote bin [px]")
    p.add_argument("--n-vote", type=int, default=300, help="brightest top-view beads used for the translation vote")
    p.add_argument("--n-boot", type=int, default=200, help="bootstrap resamples for the transform error")
    p.add_argument("--px-nm", type=float, default=None, help="sample-plane pixel size [nm], to also report nm")
    args = p.parse_args()

    stack, file = load_stack(args.stack)
    out = args.out or DEFAULT_OUT / (args.stack.name if args.stack.is_dir() else file.parent.name)
    out.mkdir(parents=True, exist_ok=True)
    H = args.split if args.split is not None else detect_split(stack)
    print(f"{file.name}: {len(stack)} frames, split row {H}" + ("" if args.split is not None else " (auto-detected)"))

    top_img, z_top, e_top = focus_image(stack, slice(0, H), args.nz)
    bot_img, z_bot, e_bot = focus_image(stack, slice(H, 2 * H), args.nz)
    print(f"best focus slice: top z={z_top}, bottom z={z_bot} (offset {z_bot - z_top} slices)")

    pts, snrs = {}, {}
    for name, img in (("top", top_img), ("bottom", bot_img)):
        cand, snr = detect(img, args.sigma, args.k, min_sep=4)
        sub, keep = gauss_refine(img, cand)
        pts[name], snrs[name] = sub, snr[keep]
        print(f"{name}: {len(cand)} candidates, {len(sub)} refined beads")
    top, bot = pts["top"], pts["bottom"]

    # coarse translation from the brightest top beads (the gated view only shows the brightest),
    # then an affine refined on mutual nearest neighbours with a shrinking radius
    bright = np.argsort(snrs["top"])[::-1][:args.n_vote]
    t0, prominence = vote_translation(top[bright], bot, args.vote_bin)
    print(f"voted translation {t0.round(2)}, peak / next peak = {prominence:.1f} (>~3 means unambiguous)")
    T = AffineTransform(translation=t0)
    for r in (4 * args.vote_bin, 2 * args.vote_bin, 2 * args.radius, args.radius):
        i, j = mutual_nn(T(top), bot, r)
        T = fit_affine(top[i], bot[j])
    src, dst = top[i], bot[j]
    print(f"{len(src)} matched pairs")

    _, inl = ransac((src, dst), AffineTransform, min_samples=3, residual_threshold=2.0, max_trials=2000)
    src, dst = src[inl], dst[inl]
    T = fit_affine(src, dst)
    res = T(src) - dst
    fit = float(np.sqrt(np.mean(np.sum(res ** 2, 1))))
    cv = cv_rms(src, dst)
    print(f"{inl.sum()} RANSAC inliers")

    # error of the transform itself over the bead field, by bootstrapping the bead pairs;
    # the CV residual also contains the beads' own localisation noise
    x0, x1 = np.percentile(src[:, 0], [1, 99])
    y0, y1 = np.percentile(src[:, 1], [1, 99])
    gx, gy = np.meshgrid(np.linspace(x0, x1, 25), np.linspace(y0, y1, 25))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    rng = np.random.default_rng(1)
    maps = [fit_affine(src[k], dst[k])(grid) for k in (rng.integers(0, len(src), len(src)) for _ in range(args.n_boot))]
    sd = np.sqrt(np.var(np.stack(maps), axis=0).sum(1))
    tre_rms, tre_max = float(np.sqrt(np.mean(sd ** 2))), float(sd.max())

    nm = (lambda v: f" ({v * args.px_nm:.1f} nm)") if args.px_nm else (lambda v: "")
    warp = T.params[:2]
    lp = linear_params(warp[:, :2])
    print(f"\nfit RMS {fit:.3f} px, 5-fold CV RMS {cv:.3f} px{nm(cv)}")
    print(f"transform error RMS / max over field: {tre_rms:.3f}{nm(tre_rms)} / {tre_max:.3f}{nm(tre_max)} px")
    print(f"per-bead localisation noise implied by CV (split equally between views): ~{cv / np.sqrt(2):.3f} px")
    print("  " + "  ".join(f"{k}={v:.5f}" for k, v in lp.items()) + f"  translation={warp[:, 2].round(2)}")

    result = dict(stack=str(file), split=int(H), warp=warp.tolist(), focus_z=dict(top=z_top, bottom=z_bot),
                  n_pairs=int(len(src)), fit_rms=fit, cv_rms=cv, tre_rms=tre_rms, tre_max=tre_max, **lp,
                  translation=warp[:, 2].tolist(),
                  note="warp maps top (x,y) -> bottom-local (x,y), y from the split row; use as "
                       "cv2.warpAffine(bottom_view, warp, (W, split), flags=WARP_INVERSE_MAP)")
    (out / "bead_registration.json").write_text(json.dumps(result, indent=2))

    # ------------------------------------------------------------- figure
    fig, ax = plt.subplots(2, 2, figsize=(16, 13))
    for a, img, det, mat, z, label in ((ax[0, 0], top_img, top, src, z_top, "top"),
                                       (ax[0, 1], bot_img, bot, dst, z_bot, "bottom")):
        a.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.9))
        a.plot(det[:, 0], det[:, 1], "o", mfc="none", mec="y", ms=6, mew=0.6, label="detected")
        a.plot(mat[:, 0], mat[:, 1], "+", c="c", ms=6, label="matched")
        a.set_title(f"{label} view (z={z}), {len(det)} beads")
    ax[0, 0].legend(fontsize=8)
    a = ax[1, 0]
    a.plot(e_top / e_top.max(), label="top")
    a.plot(e_bot / e_bot.max(), label="bottom")
    a.set_xlabel("z slice")
    a.set_ylabel("bead DoG peak (norm.)")
    a.set_title("focus curves")
    a.legend()
    a = ax[1, 1]
    q = a.quiver(src[:, 0], src[:, 1], res[:, 0], -res[:, 1], np.linalg.norm(res, axis=1), angles="xy",
                 scale_units="xy", scale=0.01, cmap="viridis")
    a.invert_yaxis()
    a.set_aspect("equal")
    fig.colorbar(q, ax=a, fraction=0.03, label="|residual| [px]")
    a.set_title(f"affine residuals (x100), CV RMS {cv:.2f} px, {len(src)} beads")
    fig.tight_layout()
    fig.savefig(out / "bead_registration.png", dpi=70)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
