"""
Compare lifetime estimators per pixel (bulk stacks) or per molecule (sm_lifetime.py output), and their
spread against the spread expected from photon statistics.

Estimators (first harmonic, relative to the IRF at the same pixel)
------------------------------------------------------------------
U_phase, U_mod, G_phase, G_mod  phasor of one view's offset-subtracted trace. Not normalised, so excitation
                                drifts between frames (e.g. the laser settling after the shutter opens) and
                                blinking enter; they shift U and G phases in opposite directions.
mean_phase, mean_mod            phasor of the mean of the U and G phasors.
r_phase, r_mod                  phasor of r_k = G_k / (G_k + U_k). Normalising each frame by U + G removes
                                anything common to both views; normalised U is 1 - r, so it carries the same
                                phase, and r is the only independent quantity left.
ratio                           weighted joint fit of r_k to the LUT curves (as sm_lifetime.py).

Expected spread
---------------
Every pixel / molecule is replaced by a single-exponential sample at the data set's median r phase lifetime,
with an unmodulated fraction that reproduces its median r modulation, and the measured per-frame totals
U_k + G_k (laser drift and blinking kept). Noise is added and the estimators rerun:
  shot      Poisson noise of the detected photons (--e-per-adu)
  measured  bulk: shot noise x --excess (camera noise from a photon-transfer curve);
            molecules: the per-frame ratio errors sr of sm_lifetime.py (background, read noise, fit)
The spread is the IQR over pixels / molecules. What the simulation leaves out: real heterogeneity,
spatial calibration errors and any noise not in the model.

Outputs (in --out): estimators_box.png, estimators_hist.png, estimators_spread.png, estimators_summary.csv

Usage
-----
    conda run -n image python lifetime_estimators.py --irf <irf_registered.tif> --lut <irf_lut.npz>
        --bulk <registered.tif ...> --mol <molecules.csv ...> --out <dir>
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tifffile

import lifetime_lut as ll
import modulation as md
import sm_lifetime as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KEYS = ["U_phase", "U_mod", "G_phase", "G_mod", "mean_phase", "mean_mod", "r_phase", "r_mod", "ratio"]
TAUS = np.arange(0, 12.0001, 0.02)


class Model:
    """Phasor and LUT helpers for one calibration."""

    def __init__(self, lut_path, phase_step):
        self.L = sm.load_lut(lut_path)
        self.per, self.sign, self.omega = self.L["period"], self.L["model"]["sign"], self.L["model"]["omega"]
        self.phases = np.arange(6) * phase_step
        self.x = 2 * np.pi * self.phases / self.per
        self.P = np.linalg.pinv(md.design(self.phases, self.per, 2))

    def h1(self, trace):
        """First-harmonic phasor c1/c0 of traces (n_phase, n)."""
        c = self.P @ trace
        return (c[1] - 1j * c[2]) / c[0]

    def lifetimes(self, h):
        """(phase, modulation) lifetimes [ns] of IRF-relative phasors."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return (-self.sign * np.tan(np.angle(h)) / self.omega,
                    np.sqrt(np.clip(1 / np.abs(h) ** 2 - 1, 0, None)) / self.omega)

    def curves(self, cU, cG):
        """LUT curves R_k(tau) (n_phase, n_tau, n) from IRF harmonic coefficients."""
        CU, CG = ll.harmonic_coef(cU), ll.harmonic_coef(cG)
        return np.stack([ll.lut(CU, CG, x, TAUS, self.L["model"]) for x in self.x])

    def estimates(self, u, g, ui, gi, w, Rs):
        """All estimators from traces u, g and the IRF traces ui, gi (n_phase, n)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            r, ri = g / (u + g), gi / (ui + gi)
        hu, hg, hr = self.h1(u) / self.h1(ui), self.h1(g) / self.h1(gi), self.h1(r) / self.h1(ri)
        e = {}
        for k, h in (("U", hu), ("G", hg), ("mean", (hu + hg) / 2), ("r", hr)):
            e[f"{k}_phase"], e[f"{k}_mod"] = self.lifetimes(h)
        e["ratio"] = ll.joint_fit(r, Rs, TAUS, w)[0]
        return e


def iqr(v):
    v = v[np.isfinite(v)]
    return np.subtract(*np.percentile(v, [75, 25])) if len(v) else np.nan


def bulk_sample(path, M, Ui, Gi):
    """Offset-subtracted binned U, G traces of the calibrated pixels, IRF traces there, and LUT curves."""
    L = M.L
    reg = tifffile.imread(path).astype(np.float32)
    U, G = reg[:, 0], reg[:, 1]
    bg = md.background(U, 250)
    U, G = md.bin_stack(U - bg[:, None, None], L["bin"]), md.bin_stack(G - bg[:, None, None], L["bin"])
    N = U + G
    m = L["mask"] & np.isfinite(N).all(0) & np.isfinite(Ui + Gi).all(0)
    m &= N.mean(0) > 0.3 * np.nanpercentile(N.mean(0)[m], 99)
    return U[:, m], G[:, m], Ui[:, m], Gi[:, m], M.curves(L["cU"][:, m], L["cG"][:, m])


def mol_sample(path, M, Ui, Gi):
    """Per-frame counts of the kept molecules, IRF traces at their pixel, LUT curves and ratio errors."""
    L, ph = M.L, M.phases
    d = pd.read_csv(path)
    d = d[d.good.astype(bool)]
    u, g, sr = (d[[f"{c}_{p:g}" for p in ph]].to_numpy().T for c in ("NU", "NG", "sr"))
    ok = np.isfinite(u).all(0) & np.isfinite(g).all(0) & np.isfinite(sr).all(0) & ((u + g) > 0).all(0)
    d, u, g, sr = d[ok], u[:, ok], g[:, ok], sr[:, ok]
    by = np.clip((d.y.to_numpy() // L["bin"]).astype(int), 0, L["mask"].shape[0] - 1)
    bx = np.clip((d.x.to_numpy() // L["bin"]).astype(int), 0, L["mask"].shape[1] - 1)
    ny, nx = L["idx"][0][by, bx], L["idx"][1][by, bx]
    return u, g, Ui[:, ny, nx], Gi[:, ny, nx], M.curves(L["cU"][:, ny, nx], L["cG"][:, ny, nx]), sr


def simulate(M, u, g, ui, gi, Rs, e_obs, photons_per_adu, rng, extra_var=None):
    """Noisy single-exponential copies of the sample: (estimates at shot noise, at measured noise)."""
    tau0 = np.nanmedian(e_obs["r_phase"])
    tm = np.nanmedian(e_obs["r_mod"])
    k0 = np.clip(np.searchsorted(TAUS, tau0), 0, len(TAUS) - 1)
    H = 1 / np.sqrt(1 + (M.omega * TAUS[k0]) ** 2)
    f0 = float(np.clip(1 - (1 / np.sqrt(1 + (M.omega * tm) ** 2)) / H, -0.3, 0.9))
    with np.errstate(invalid="ignore", divide="ignore"):
        rbar = (gi / (ui + gi)).mean(0)
    r_true = (1 - f0) * Rs[:, k0] + f0 * rbar
    N = u + g
    out = []
    for var_scale in (None, extra_var):
        ne = np.clip(N, 0, None) * photons_per_adu
        lam_u, lam_g = ne * (1 - r_true), ne * r_true
        if var_scale is None:
            us, gs = rng.poisson(lam_u), rng.poisson(lam_g)
        else:  # Gaussian with the measured variance scale
            us = lam_u + rng.normal(size=lam_u.shape) * np.sqrt(lam_u * var_scale)
            gs = lam_g + rng.normal(size=lam_g.shape) * np.sqrt(lam_g * var_scale)
        us, gs = us / photons_per_adu, gs / photons_per_adu
        with np.errstate(invalid="ignore", divide="ignore"):
            r = gs / (us + gs)
            w = (us + gs) / np.clip(r * (1 - r), 1e-3, None)
        out.append(M.estimates(us, gs, ui, gi, w, Rs))
    return out, tau0, f0


def short(name):
    return name.replace("15,3W_1,44Vrms_", "").replace("_20261022_", " ")[-46:]


def colour(k):
    return "C2" if k == "ratio" else ("C0" if k.endswith("phase") else "C3")


def plot_distributions(samples, kind_plot, out, irf_name):
    nc = 4
    nr = int(np.ceil(len(samples) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(5.2 * nc, 4.3 * nr), squeeze=False)
    for a, s in zip(axs.flat, samples):
        data = [np.clip(s["obs"][k][np.isfinite(s["obs"][k])], -2, 14) for k in KEYS]
        if kind_plot == "box":
            bp = a.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.6)
            for patch, k in zip(bp["boxes"], KEYS):
                patch.set_facecolor(colour(k))
                patch.set_alpha(0.45)
            for med in bp["medians"]:
                med.set_color("k")
            a.set_xticks(range(1, len(KEYS) + 1), [k.replace("_", "\n") for k in KEYS], fontsize=7)
            a.set_ylim(-1, 10)
            a.set_ylabel("tau [ns]")
        else:
            bins = np.arange(-1, 10, 0.1 if s["kind"] == "bulk" else 0.25)
            for k, v in zip(KEYS, data):
                if k.startswith("mean"):
                    continue
                a.hist(v, bins, histtype="step", lw=1.4, density=True, ls="--" if k.startswith("G") else "-",
                       color="k" if k.startswith("r_") else colour(k), label=f"{k} {np.median(v):.2f}")
            a.set_xlabel("tau [ns]")
            a.legend(fontsize=6)
        a.grid(alpha=0.3)
        a.set_title(f"{s['kind']}: {short(s['name'])}  (n={len(s['obs']['ratio'])})", fontsize=8)
    for a in list(axs.flat)[len(samples):]:
        a.axis("off")
    fig.suptitle(f"lifetime estimators vs IRF {irf_name[:60]} (first harmonic; whiskers 5-95 %)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"estimators_{kind_plot}.png", dpi=75)
    plt.close(fig)


def plot_spread(samples, out):
    nc = 4
    nr = int(np.ceil(len(samples) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(5.2 * nc, 4.0 * nr), squeeze=False)
    xk = np.arange(len(KEYS))
    for a, s in zip(axs.flat, samples):
        obs = [iqr(s["obs"][k]) for k in KEYS]
        a.bar(xk, obs, 0.6, color=[colour(k) for k in KEYS], alpha=0.45, label="observed")
        a.plot(xk, [iqr(s["shot"][k]) for k in KEYS], "kv", ms=7, label="expected: shot noise")
        a.plot(xk, [iqr(s["meas"][k]) for k in KEYS], "k^", mfc="none", ms=7, label="expected: measured noise")
        a.set_yscale("log")
        a.set_ylim(0.01, 20)
        a.set_xticks(xk, [k.replace("_", "\n") for k in KEYS], fontsize=7)
        a.set_ylabel("IQR of tau [ns]")
        a.grid(alpha=0.3, which="both")
        a.set_title(f"{s['kind']}: {short(s['name'])}\nmodel tau {s['tau0']:.2f} ns, unmodulated {s['f0']:+.2f}",
                    fontsize=8)
    list(axs.flat)[0].legend(fontsize=7, loc="upper left")
    for a in list(axs.flat)[len(samples):]:
        a.axis("off")
    fig.suptitle("spread of each estimator (IQR over pixels / molecules) vs simulation of a single-exponential "
                 "sample with the same counts", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "estimators_spread.png", dpi=75)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--irf", type=Path, required=True, help="registered IRF stack (methylene blue)")
    p.add_argument("--lut", type=Path, required=True, help="<cal>_lut.npz of that IRF from lifetime_lut.py")
    p.add_argument("--bulk", type=Path, nargs="*", default=[], help="registered bulk stacks (per-pixel estimates)")
    p.add_argument("--mol", type=Path, nargs="*", default=[], help="<name>_molecules.csv from sm_lifetime.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase-step", type=float, default=30.0, help="RF phase step between frames [deg]")
    p.add_argument("--e-per-adu", type=float, default=0.217, help="camera conversion [photoelectrons / ADU]")
    p.add_argument("--excess", type=float, default=1.45,
                   help="camera variance / shot-noise variance for bulk pixels (photon-transfer curve)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    M = Model(args.lut, args.phase_step)
    b = M.L["bin"]
    reg = tifffile.imread(args.irf).astype(np.float32)
    bg = md.background(reg[:, 0], 250)
    Ui, Gi = (md.bin_stack(reg[:, c] - bg[:, None, None], b) for c in (0, 1))

    samples = []
    for path in args.bulk:
        u, g, ui, gi, Rs = bulk_sample(path, M, Ui, Gi)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = g / (u + g)
        obs = M.estimates(u, g, ui, gi, (u + g) / np.clip(r * (1 - r), 1e-3, None), Rs)
        # a binned pixel is the mean of b^2 camera pixels
        (shot, meas), tau0, f0 = simulate(M, u, g, ui, gi, Rs, obs, args.e_per_adu * b * b, rng, args.excess)
        samples.append(dict(kind="bulk", name=path.name.removesuffix("_registered.tif"), obs=obs, shot=shot,
                            meas=meas, tau0=tau0, f0=f0))
    for path in args.mol:
        u, g, ui, gi, Rs, sr = mol_sample(path, M, Ui, Gi)
        obs = M.estimates(u, g, ui, gi, 1 / np.clip(sr, 1e-3, None) ** 2, Rs)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = g / (u + g)
            ne = (u + g) * args.e_per_adu
            excess = (sr / np.sqrt(np.clip(r * (1 - r), 1e-4, None) / ne)) ** 2   # measured / binomial variance
        (shot, meas), tau0, f0 = simulate(M, u, g, ui, gi, Rs, obs, args.e_per_adu, rng, np.clip(excess, 1, 100))
        samples.append(dict(kind="bead" if "beads" in path.name else "smlm",
                            name=path.name.removesuffix("_molecules.csv"), obs=obs, shot=shot, meas=meas,
                            tau0=tau0, f0=f0))

    rows = []
    for s in samples:
        row = dict(kind=s["kind"], name=s["name"], n=len(s["obs"]["ratio"]), model_tau=s["tau0"], model_f=s["f0"])
        for k in KEYS:
            row[k] = np.nanmedian(s["obs"][k])
            row[f"{k}_iqr"] = iqr(s["obs"][k])
            row[f"{k}_iqr_shot"] = iqr(s["shot"][k])
            row[f"{k}_iqr_meas"] = iqr(s["meas"][k])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "estimators_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        t = df.assign(name=df.name.map(short))
        print("medians [ns]")
        print(t[["kind", "name", "n"] + KEYS].round(2).to_string(index=False))
        print("\nIQR observed / expected (shot) / expected (measured noise) [ns]")
        for _, r in t.iterrows():
            print(f"{r.kind:4s} {r['name']:46s} " + "  ".join(
                f"{k} {r[k + '_iqr']:.2f}/{r[k + '_iqr_shot']:.2f}/{r[k + '_iqr_meas']:.2f}"
                for k in ("r_phase", "r_mod", "ratio", "U_phase", "G_phase")))

    for kind_plot in ("box", "hist"):
        plot_distributions(samples, kind_plot, args.out, args.irf.name)
    plot_spread(samples, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
