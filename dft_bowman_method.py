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
    p.add_argument("--irf", nargs="*", default=[DEFAULT_IRF],
                   help="full path to registered IRF data")
    p.add_argument("--out", type=Path)
    p.add_argument("--freq", type=float, default=20.0, help="RF frequency [MHz] of laser")
    p.add_argument("--phase-step", type=float, default=30.0, help="RF phase step between frames [deg]")
    p.add_argument("--period", type=float, default=180.0, help="gating period in RF phase [deg]")
    p.add_argument("--harmonics", type=int, default=2, help="max harmonics in the fit (capped by #phases)")
    p.add_argument("--bin", type=int, default=8, help="spatial binning for the maps")
    p.add_argument("--roi", type=int, nargs=2, default=[200,200], help="ROI dimensions [Y, X]")
    p.add_argument("--roi-center", type=int, nargs=2, default=None, metavar=("Y", "X"),
                   help="ROI centre [full-res px] (default: centroid of the mask)")
    p.add_argument("--bg", type=float, default=100, help="fixed camera offset (default: from image margins)")
    p.add_argument("--use-g", action="store_true")
    p.add_argument("--show-plots", action="store_false")
    p.add_argument("--duty-cycle", type=float, default=0.5)
    p.add_argument("--use-average-roi-irf", action="store_true")
    args = p.parse_args()
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)

    print("run FT IRF")

    # subtract digital offset from the IRF data (float first: uint16 - 100 wraps below 0)
    if args.use_average_roi_irf:
        data = average_roi_irf(args.irf[0], args.roi, bg=args.bg)
    else:
        data = imread(args.irf[0]).astype(np.float32) - args.bg
    
    dft_irf = ft_data(data, args.phase_step, args.period, bg=args.bg)
    
    # make sure when single tif is apssed as data that the string does not get treated as a list
    if len(args.data) == 1 and isinstance(args.data[0], str):
        args.data = [Path(args.data[0])]
    if len(args.data) == 1 and args.data[0].is_dir():
        args.data = list(args.data[0].glob("*_registered.tif"))

    dft_data = [ft_data(imread(path).astype(np.float32) - args.bg,
                args.phase_step, args.period, bg=args.bg) for path in args.data]

    if args.roi_center is None:
        args.roi_center = [dft_irf.shape[0] // 2, dft_irf.shape[1] // 2]
    roi_rect = [args.roi_center[0] - args.roi[0] // 2, args.roi_center[0] + args.roi[0] // 2,
                args.roi_center[1] - args.roi[1] // 2, args.roi_center[1] + args.roi[1] // 2]
    y_start, y_end, x_start, x_end = roi_rect
    dft_irf = dft_irf[y_start:y_end, x_start:x_end]
    dft_data = [dft[y_start:y_end, x_start:x_end] / dft_irf for dft in dft_data]
    data_labels = [path.name for path in args.data]
    print(dft_irf.shape)

    if args.out is not None:
        # make phasor data subfolder output dir args.out/data
        args.out.mkdir(parents=True, exist_ok=True)
        data_dir = args.out / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for n, dft in enumerate(dft_data):
            np.save(data_dir / f"{data_labels[n]}.npy", dft)

    # phasor is at the gate frequency: 360/period gate cycles per RF cycle (40 MHz for 20 MHz RF, 180 deg)
    plot_phasors(dft_data, data_labels, args.freq * 360 / args.period, 
                 save_phasors=args.out, show_plots=args.show_plots)

def average_roi_irf(path, roi_rect, bg=100):
    data = imread(path).astype(np.float32)  # is a stack of length n_phase
    # subtract digital offset from the IRF data (float first: uint16 - 100 wraps below 0)
    data = data - bg
    y_start, y_end, x_start, x_end = roi_rect
    # add the bg back
    data = data + bg 
    return data[:, y_start:y_end, x_start:x_end].mean(axis=(1, 2))

def plot_phasors(dft_data, data_labels, freq, save_phasors=None, show_plots=False):
    # calculate point on g,s circle for 1, 2, 3, 5, 10ns single exp
    omega = 2*np.pi*freq*1e6
    tau = np.array([1e-9, 2e-9, 3e-9, 5e-9, 10e-9])
    g_markers = 1/(1 + (omega*tau)**2)
    s_markers = g_markers * omega*tau
    for n, dft in enumerate(dft_data):
        plt.figure()
        dft = dft[np.isfinite(dft)]  # histogram2d needs 1-D input without NaN/inf
        g = dft.real
        s = dft.imag
        # plt.scatter(g, s, s=1)
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
        plt.title(data_labels[n])
        if save_phasors is not None:
            plt.savefig(save_phasors / f"{data_labels[n]}.png")
    if show_plots:
        plt.show()

def ft_data(data, phase_step, period, gate_idx=1, bg=100):
    """DFT of the IRF data."""
    
    # 1 = g, 0 = u
    g_over_gplusu = data[:,gate_idx,...] / (data[:,1,...] + data[:,0,...])

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

