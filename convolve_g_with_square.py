# visualize g(t) = sin^2(V/Vpi*pi/2*sin(omega*t-phi)) convolved with square wave 50% duty cycle and frequency 2*omega
# doing it in Fourier space is easier, since convolution becomes multiplication and the square wave has a simple Fourier series.
"""
Fourier series, in gating phase y = 2 omega t (the gate repeats every half RF period):

    g(y) = sin^2(Gamma/2 sin(y/2 - phi)),  Gamma = pi V/V_pi
         = (1 - J0(Gamma))/2 - sum_{n>=1} J_2n(Gamma) cos(n (y - 2 phi))        (Jacobi-Anger)
    e(y) = square wave, duty d, centred on y = 0
         = d + sum_{n>=1} 2 d sinc(n d) cos(n y)                                (sinc(x) = sin(pi x)/(pi x))

Circular convolution (1/2pi) int g(y') e(y - y') dy' multiplies the coefficients (cos x cos -> 1/2),
normalised by the mean excitation d:

    g_sq(y) = (1 - J0(Gamma))/2 - sum_n sinc(n d) J_2n(Gamma) cos(n (y - 2 phi))

For d = 0.5, sinc(n/2) = 0 for every even n: the square wave removes all even gating harmonics,
and the odd ones are scaled by 2/(pi n) (+-). The script checks the closed form against a
brute-force FFT convolution of the sampled waveforms and plots both.

Usage:  conda run -n image python convolve_g_with_square.py [--v 0.5 1.0 1.5] [--duty 0.5] [--phi 0]
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy.special import jv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N = 4096                                         # samples per gating period
NH = 40                                          # harmonics in the analytic series
y = np.arange(N) * 2 * np.pi / N                 # gating phase [rad]


def gate(v, phi):
    """g(y) sampled, y = 2 omega t; phi is the RF phase offset [rad]."""
    return np.sin(v * np.pi / 2 * np.sin(y / 2 - phi)) ** 2


def square(duty):
    """50%-duty (or other) square wave centred on y = 0, one gating period."""
    return (((y + duty * np.pi) % (2 * np.pi)) < 2 * np.pi * duty).astype(float)


def gate_coef(v):
    """Cosine-series coefficients of g: g = c0 + sum_n c_n cos(n (y - 2 phi)), n = 1..NH."""
    n = np.arange(1, NH + 1)
    return (1 - jv(0, v * np.pi)) / 2, -jv(2 * n, v * np.pi)


def square_coef(duty):
    """e = d + sum_n s_n cos(n y), normalised by the mean d: returns 1, s_n / d."""
    n = np.arange(1, NH + 1)
    return 1.0, 2 * np.sinc(n * duty)


def analytic(v, phi, duty):
    """g convolved with e / mean(e), from the product of the Fourier coefficients."""
    c0, cn = gate_coef(v)
    _, sn = square_coef(duty)
    n = np.arange(1, NH + 1)
    return c0 + (cn * sn / 2) @ np.cos(np.outer(n, y - 2 * phi))


def numeric(v, phi, duty):
    """Brute force: circular convolution of the sampled waveforms via FFT."""
    e = square(duty)
    return np.fft.ifft(np.fft.fft(gate(v, phi)) * np.fft.fft(e)).real / e.sum()


def m1(trace):
    """Fundamental modulation depth: 2 |F1| / F0."""
    F = np.fft.rfft(trace)
    return 2 * abs(F[1]) / F[0].real


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v", type=float, nargs="+", default=[0.5, 1.0, 1.5], help="V / V_pi values to plot")
    p.add_argument("--duty", type=float, default=0.5)
    p.add_argument("--phi", type=float, default=0.0, help="RF phase offset of the gate [deg]")
    p.add_argument("--out", type=Path, default=Path(r"E:\20261019\20261019_analysis\gating_model\convolve_g_with_square.png"))
    args = p.parse_args()
    phi = np.radians(args.phi)
    deg = np.degrees(y) / 2                      # gating phase in RF degrees (0-180)

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    a = ax[0, 0]
    for k, v in enumerate(args.v):
        a.plot(deg, gate(v, phi), c=f"C{k}", label=f"g(t), V/V$_\\pi$ = {v:g}")
    a.fill_between(deg, 0, square(args.duty), color="0.5", alpha=0.25, step="mid", label=f"square, duty {args.duty:g}")
    a.set_xlabel("RF phase $\\omega t$ [deg] (one gating period)")
    a.set_title("gate and excitation")
    a.legend(fontsize=8)

    a = ax[0, 1]
    print(f"duty {args.duty:g}, phi {args.phi:g} deg")
    for k, v in enumerate(args.v):
        an, nu = analytic(v, phi, args.duty), numeric(v, phi, args.duty)
        a.plot(deg, gate(v, phi), c=f"C{k}", lw=0.8, ls=":")
        a.plot(deg, an, c=f"C{k}", lw=2, label=f"V/V$_\\pi$ = {v:g}: m1 {m1(gate(v, phi)):.2f} -> {m1(an):.2f}")
        a.plot(deg[::128], nu[::128], "o", c=f"C{k}", ms=4, mfc="none")
        print(f"  V/V_pi {v:g}: max |analytic - FFT| = {np.abs(an - nu).max():.1e}   m1 gate {m1(gate(v, phi)):.3f}"
              f" -> convolved {m1(an):.3f}")
    a.set_xlabel("delay [RF deg]")
    a.set_title("g convolved with the square (line: Fourier, circles: FFT, dotted: g alone)")
    a.legend(fontsize=8)

    a = ax[1, 0]
    v = args.v[len(args.v) // 2]
    n = np.arange(1, NH + 1)
    _, cn = gate_coef(v)
    _, sn = square_coef(args.duty)
    sel = n <= 12
    a.semilogy(n[sel] - 0.15, abs(cn[sel]) + 1e-12, "o", label="|g_n| (Bessel J$_{2n}$)")
    a.semilogy(n[sel], abs(sn[sel] / 2) + 1e-12, "s", label="|e_n| (sinc)")
    a.semilogy(n[sel] + 0.15, abs(cn[sel] * sn[sel] / 2) + 1e-12, "^", label="|g_n e_n| (product)")
    a.set_ylim(1e-6, 1)
    a.set_xlabel("gating harmonic n")
    a.set_title(f"Fourier coefficients, V/V$_\\pi$ = {v:g}, duty {args.duty:g}")
    a.legend(fontsize=8)

    a = ax[1, 1]
    vs = np.linspace(0.02, 2, 200)
    a.plot(vs, [m1(gate(x, phi)) for x in vs], "k:", label="g alone (pulse excitation)")
    for d, c in ((args.duty, "C3"), (0.25, "C1"), (0.1, "C2")):
        a.plot(vs, [m1(analytic(x, phi, d)) for x in vs], c=c, label=f"g convolved with square, duty {d:g}")
    a.set_xlabel("V / V$_\\pi$")
    a.set_ylabel("m1")
    a.set_title("fundamental modulation depth")
    a.legend(fontsize=8)
    for x in ax.flat:
        x.grid(alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=80)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
