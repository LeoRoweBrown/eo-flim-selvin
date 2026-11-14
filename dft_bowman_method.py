import numpy as np
import argparse
from pathlib import Path
from tifffile import imread
from matplotlib import pyplot as plt

DEFAULT_IRF = "E:\\20261106\\analysis\\registered\\vin_0.12_15,48W_1,7Vrms_MB_100uM_uM_od0,5_z0um_0,5s_laser_settle_3773tirf_25pc_duty_10mW_19.997ms_20261106_105308_registered.tif"
DEFAULT_DATA = "E:\\20261106\\analysis\\registered\\vin_0.12_15,48W_1,7Vrms_atto_az_tween_1,25uM_uM_od0,5_z0um_0,5s_laser_settle_3773tirf_25pc_duty_10mW_40ms_20261106_110850_registered.tif"
DEFAULT_DATA = "E:\\20261106\\analysis\\registered\\vin_0.12_15,48W_1,7Vrms_MB_100uM_uM_od0,5_z0um_0,5s_laser_settle_3773tirf_25pc_duty_10mW_40ms_20261106_110402_registered.tif"

DEFAULT_OUT = "OUT"

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, nargs="*", default=[DEFAULT_DATA],
                   help="folder of *_registered.tif and/or individual files")
    p.add_argument("--irf", nargs="*", default=[DEFAULT_IRF], type=Path,
                   help="full path to registered IRF data")
    p.add_argument("--out", type=Path)
    p.add_argument("--freq", type=float, default=40.0, help="RF frequency [MHz] of laser")
    p.add_argument("--phase-step", type=float, default=30.0, help="RF phase step between frames [deg]")
    p.add_argument("--period", type=float, default=180.0, help="gating period in RF phase [deg]")
    p.add_argument("--harmonics", type=int, default=2, help="max harmonics in the fit (capped by #phases)")
    p.add_argument("--bin", type=int, default=8, help="spatial binning for the maps")
    p.add_argument("--roi", type=int, nargs=2, default=[200,200], help="ROI dimensions [Y, X]")
    p.add_argument("--roi-center", type=int, nargs=2, default=None, metavar=("Y", "X"),
                   help="ROI centre [full-res px] (default: centroid of the mask)")
    p.add_argument("--bg", type=float, default=100, help="fixed camera offset (default: from image margins)")
    p.add_argument("--use-g", action="store_true")
    p.add_argument("--hide-plots", action="store_true")
    p.add_argument("--duty-cycle", type=float, default=0.5)
    p.add_argument("--use-average-roi-irf", action="store_true")
    args = p.parse_args()

    # data files: a folder -> every *_registered.tif in it
    files = []
    for d in args.data:
        d = Path(d)
        files += sorted(d.glob("*_registered.tif")) if d.is_dir() else [d]

    # output: <out>/<irf name>/ for the pngs, <out>/<irf name>/data/ for the .npy and args.txt
    save_path = data_dir = None
    if args.out is not None:
        save_path = args.out / args.irf[0].stem
        data_dir = save_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        # save args first, so they are there even if a dataset fails later
        (data_dir / "args.txt").write_text("".join(f"{k}: {v}\n" for k, v in vars(args).items()))

    # IRF: full-image DFT, then cropped to the ROI (or one ROI-averaged phasor)
    print(f"run FT IRF: {args.irf[0].name}")
    irf = imread(args.irf[0]).astype(np.float32) - args.bg   # float first: uint16 - 100 wraps below 0
    if args.roi_center is None:
        args.roi_center = [irf.shape[-2] // 2, irf.shape[-1] // 2]
    y_start, y_end, x_start, x_end = [args.roi_center[0] - args.roi[0] // 2, args.roi_center[0] + args.roi[0] // 2,
                                      args.roi_center[1] - args.roi[1] // 2, args.roi_center[1] + args.roi[1] // 2]
    irf = irf[..., y_start:y_end, x_start:x_end]
    if args.use_average_roi_irf:
        irf = irf.mean(axis=(-2, -1))       # (T, 2): one IRF phasor for the whole ROI
    dft_irf = ft_data(irf, args.phase_step, args.period, bg=args.bg)
    del irf

    freq = args.freq #  * 360 / args.period

    # one dataset at a time: DFT -> normalise -> save .npy and png straight away
    failed = []
    for n, path in enumerate(files, 1):
        label = path.name
        print(f"[{n}/{len(files)}] {label}")
        try:
            data = imread(path).astype(np.float32)[..., y_start:y_end, x_start:x_end] - args.bg
            dft = ft_data(data, args.phase_step, args.period, bg=args.bg) / dft_irf
            del data
            if data_dir is not None:
                np.save(data_dir / f"{label}.npy", dft)
            plot_phasor(dft, label, freq, save_path=save_path, close=args.hide_plots)
        except Exception as e:  # keep going; report at the end
            print(f"  FAILED ({type(e).__name__}: {e})")
            failed.append(label)

    if failed:
        print(f"\n{len(failed)} failed: " + ", ".join(failed))
    if save_path is not None:
        print(f"wrote {save_path}")
    if not args.hide_plots:
        plt.show()


def plot_phasor(dft, label, freq, save_path=None, close=True):
    """2D histogram of one dataset's phasors with the universal circle; saved as <label>.png."""
    # calculate point on g,s circle for 1, 2, 3, 5, 10ns single exp
    omega = 2*np.pi*freq*1e6
    tau = np.array([1e-9, 2e-9, 3e-9, 5e-9, 10e-9])
    g_markers = 1/(1 + (omega*tau)**2)
    s_markers = g_markers * omega*tau
    plt.figure()
    dft = dft[np.isfinite(dft)]  # histogram2d needs 1-D input without NaN/inf
    g = dft.real
    s = dft.imag
    heatmap, edgex, edgey = np.histogram2d(g, s, bins=200, range=[[-0.1, 1.2], [-0.1, 0.6]])
    plt.pcolormesh(edgex, edgey, heatmap.T, cmap="magma")

    # now plot universal circle (centre (0.5, 0), radius 0.5)
    theta = np.linspace(0, np.pi, 100)
    plt.plot(0.5 + 0.5*np.cos(theta), 0.5*np.sin(theta), "w")
    plt.scatter(g_markers, s_markers, c="w")
    for g_marker, s_marker, lifetime in zip(g_markers, s_markers, tau):
        plt.text(g_marker, s_marker, f"{lifetime * 1e9:.0f} ns", color='w')
    plt.xlim(-0.1, 1.2)
    plt.ylim(-0.1, 0.6)
    plt.colorbar()
    plt.title(label, fontsize=8)
    if save_path is not None:
        plt.savefig(save_path / f"{label}.png")
    if close:
        plt.close()

def ft_data(data, phase_step, period, gate_idx=1, bg=100, threshold=0):
    """DFT of the IRF data."""
    
    # 1 = g, 0 = u
    g_over_gplusu = data[:,gate_idx,...] / (data[:,1,...] + data[:,0,...])
    g_mask = data[:, gate_idx, ...] < threshold
    u_mask = data[:, 0, ...] < threshold
    # AND operation on two masks
    g_and_u_mask = g_mask | u_mask
    g_over_gplusu[g_and_u_mask] = np.nan

    # get phases
    phases = np.array([phase_step*i for i in range(data.shape[0])])
    k = np.exp(1j * 2 * np.pi * phases / period)
    print("phases size", phases.size)
    print(phases)
    # make phases broadcastable to the IRF data shape, match trailing dimensions
    g_over_gplusu = np.moveaxis(g_over_gplusu, 0, -1)
    dft = np.sum(g_over_gplusu * k, axis=-1) # / np.sum(g_over_gplusu, axis=-1)

    # reverse phase list direction
    dft = np.sum(g_over_gplusu * k[::-1], axis=-1)
    print(dft.shape)
    return dft

if __name__ == "__main__":
    main()

