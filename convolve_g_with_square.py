# visualize g(t) = sin^2(V/Vpi*pi/2*sin(omega*t-phi) + delta/2) convolved with a square-wave excitation at 2*omega
# (40 MHz, the default) or omega (20 MHz). doing it in Fourier space is easier, since convolution becomes multiplication
# and the square wave has a simple Fourier series.
"""
Fourier series in RF phase theta = omega t (period 2 pi), gate offset phi, static bias delta:

    g(theta) = sin^2(Gamma/2 sin(theta - phi) + delta/2),  Gamma = pi V/V_pi
             = (1 - J0(Gamma) cos delta)/2
               - cos delta sum_{m even} J_m(Gamma) cos(m (theta - phi))
               + sin delta sum_{m odd}  J_m(Gamma) sin(m (theta - phi))          (Jacobi-Anger)

With delta = 0 only the even RF harmonics survive: the gate repeats every half RF period (40 MHz gating).
The bias delta adds the odd ones, ~ J1(Gamma) sin delta at the RF frequency itself (20 MHz gating).

    e(theta) = square wave at q x the RF frequency (q = 2: 40 MHz, q = 1: 20 MHz), duty d, centred on theta = 0
             = d + sum_{n>=1} 2 d sinc(n d) cos(n q theta)                       (sinc(x) = sin(pi x)/(pi x))

Circular convolution (1/2pi) int g(theta') e(theta - theta') dtheta' keeps only the RF harmonics m = n q and multiplies
the coefficients (x 1/2), normalised by the mean excitation d:

    g_sq(theta) = (1 - J0 cos delta)/2 + sum_n sinc(n d) [gate term at m = n q]

For d = 0.5, sinc(n/2) = 0 for every even n, and the odd n are scaled by 2/(pi n) (+-).
- 40 MHz (q = 2): the fundamental is -sinc(d) J2 cos delta: largest at delta = 0.
- 20 MHz (q = 1): the fundamental is sinc(d) J1 sin delta: largest at delta = 90 deg (quarter-wave bias). At that bias
  the gate is (1 + sin(Gamma sin theta))/2, a full swing around 0.5 with no even (40 MHz) content at all.

The script checks the closed form against a brute-force FFT convolution of the sampled waveforms and plots both.
m1 is the fundamental modulation depth 2|F1|/F0 at the excitation frequency, i.e. the IRF |h| of r = G/(G+U).

Usage:  conda run -n image python convolve_g_with_square.py [--v 0.5 1.0 1.5] [--duty 0.5] [--phi 0]
        ... --f-exc 20 --delta 90          (20 MHz excitation, quarter-wave bias)
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy.special import jv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N = 4096                                         # samples per RF period
NH = 40                                          # RF harmonics in the analytic series
th = np.arange(N) * 2 * np.pi / N                # RF phase [rad]


def gate(v, phi, delta):
    """g(theta) sampled over one RF period; phi is the RF phase offset, delta the static bias [rad]."""
    return np.sin(v * np.pi / 2 * np.sin(th - phi) + delta / 2) ** 2


def square(duty, q):
    """Square wave at q x the RF frequency, centred on theta = 0, sampled over one RF period."""
    return (((q * th + duty * np.pi) % (2 * np.pi)) < 2 * np.pi * duty).astype(float)


def gate_coef(v, delta):
    """g = c0 + sum_m a_m cos(m (theta - phi)) + b_m sin(m (theta - phi)), m = 1..NH."""
    m = np.arange(1, NH + 1)
    j = jv(m, v * np.pi)
    even = m % 2 == 0
    c0 = (1 - jv(0, v * np.pi) * np.cos(delta)) / 2
    return c0, np.where(even, -j * np.cos(delta), 0), np.where(even, 0, j * np.sin(delta))


def square_coef(duty, q):
    """e = d + sum_m s_m cos(m theta) over RF harmonics m, normalised by the mean d: returns 1, s_m."""
    m = np.arange(1, NH + 1)
    return 1.0, np.where(m % q == 0, 2 * np.sinc(m / q * duty), 0)


def analytic(v, phi, duty, delta, q):
    """g convolved with e / mean(e), from the product of the Fourier coefficients."""
    c0, a, b = gate_coef(v, delta)
    _, s = square_coef(duty, q)
    m = np.arange(1, NH + 1)
    x = np.outer(m, th - phi)
    return c0 + (a * s / 2) @ np.cos(x) + (b * s / 2) @ np.sin(x)


def numeric(v, phi, duty, delta, q):
    """Brute force: circular convolution of the sampled waveforms via FFT."""
    e = square(duty, q)
    return np.fft.ifft(np.fft.fft(gate(v, phi, delta)) * np.fft.fft(e)).real / e.sum()


def m1(trace, q):
    """Fundamental modulation depth at the excitation frequency: 2 |F_q| / F0 over one RF period."""
    F = np.fft.rfft(trace)
    return 2 * abs(F[q]) / F[0].real


def amp(trace, q):
    """Absolute fundamental amplitude of r = G/(G+U), the same for G and U = 1 - G."""
    return 2 * abs(np.fft.rfft(trace)[q]) / len(trace)


def phase_info(trace, q):
    """Photon-limited information on a delay carried by the fundamental (what the first-harmonic phasor uses), with
    phase steps spread evenly over the period and a binomial U/G split: swing^2 / mean(r (1 - r)).
    Lifetime precision scales as 1/sqrt of this."""
    return amp(trace, q) ** 2 / np.mean(trace * (1 - trace))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v", type=float, nargs="+", default=[0.5, 1.0, 1.5], help="V / V_pi values to plot")
    p.add_argument("--duty", type=float, default=0.5)
    p.add_argument("--phi", type=float, default=0.0, help="RF phase offset of the gate [deg]")
    p.add_argument("--f-exc", type=int, choices=(40, 20), default=40, help="excitation frequency [MHz], RF = 20 MHz")
    p.add_argument("--delta", type=float, default=0.0, help="static retardance bias [deg]")
    p.add_argument("--out", type=Path, default=Path(r"E:\20261019\20261019_analysis\gating_model\convolve_g_with_square.png"))
    args = p.parse_args()
    phi, delta, q = np.radians(args.phi), np.radians(args.delta), args.f_exc // 20
    period = 360 // q                            # one excitation period in RF degrees
    show = slice(0, N // q + 1)
    deg = np.degrees(th)
    tag = f"{args.f_exc} MHz excitation, delta {args.delta:g} deg"

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    a = ax[0, 0]
    for k, v in enumerate(args.v):
        g = gate(v, phi, delta)
        a.plot(deg[show], g[show], c=f"C{k}", label=f"g (G view), V/V$_\\pi$ = {v:g}")
        a.plot(deg[show], 1 - g[show], c=f"C{k}", ls="--", lw=1, label="u = 1 - g (U view)")
    a.fill_between(deg[show], 0, square(args.duty, q)[show], color="0.5", alpha=0.25, step="mid",
                   label=f"square, duty {args.duty:g}")
    a.set_xlabel(f"RF phase $\\omega t$ [deg] (one excitation period, {period} deg)")
    a.set_title(f"gate and excitation ({tag})")
    a.legend(fontsize=7, ncol=2)

    a = ax[0, 1]
    print(f"{tag}, duty {args.duty:g}, phi {args.phi:g} deg")
    for k, v in enumerate(args.v):
        g = gate(v, phi, delta)
        an, nu = analytic(v, phi, args.duty, delta, q), numeric(v, phi, args.duty, delta, q)
        a.plot(deg[show], an[show], c=f"C{k}", lw=2,
               label=f"V/V$_\\pi$ = {v:g}: m1 G {m1(g, q):.2f} -> {m1(an, q):.2f}, "
                     f"U {m1(1 - g, q):.2f} -> {m1(1 - an, q):.2f}, r swing {amp(an, q):.2f}")
        a.plot(deg[show], 1 - an[show], c=f"C{k}", lw=1, ls="--")
        a.plot(deg[show][::128], nu[show][::128], "o", c=f"C{k}", ms=4, mfc="none")
        print(f"  V/V_pi {v:g}: max |analytic - FFT| = {np.abs(an - nu).max():.1e}   m1 G {m1(g, q):.3f}"
              f" -> {m1(an, q):.3f}, U {m1(1 - g, q):.3f} -> {m1(1 - an, q):.3f}, mean r {an.mean():.3f},"
              f" r swing {amp(an, q):.3f}")
    a.set_xlabel("delay [RF deg]")
    a.set_title("G = g convolved with the square (solid; circles: FFT) and U = 1 - G (dashed)")
    a.legend(fontsize=7)

    a = ax[1, 0]
    v = args.v[len(args.v) // 2]
    m = np.arange(1, NH + 1)
    _, ga, gb = gate_coef(v, delta)
    _, s = square_coef(args.duty, q)
    gm = np.hypot(ga, gb)
    sel = m <= 12 * q
    a.semilogy(m[sel] - 0.15, gm[sel] + 1e-12, "o", label="|g_m| (Bessel J$_m$)")
    a.semilogy(m[sel], abs(s[sel] / 2) + 1e-12, "s", label="|e_m| (sinc)")
    a.semilogy(m[sel] + 0.15, abs(gm[sel] * s[sel] / 2) + 1e-12, "^", label="|g_m e_m| (product)")
    a.set_ylim(1e-6, 1)
    a.set_xlabel("RF harmonic m (excitation fundamental at m = %d)" % q)
    a.set_title(f"Fourier coefficients, V/V$_\\pi$ = {v:g}, duty {args.duty:g}")
    a.legend(fontsize=8)

    a = ax[1, 1]
    vs = np.linspace(0.02, 2, 200)
    for d, c in ((args.duty, "C3"), (0.25, "C1")):
        tr = [analytic(x, phi, d, delta, q) for x in vs]
        info = np.array([phase_info(x, q) for x in tr])
        best = vs[np.argmax(info)]
        a.plot(vs, [m1(x, q) for x in tr], c=c, label=f"duty {d:g}: m1 of G")
        a.plot(vs, [m1(1 - x, q) for x in tr], c=c, ls="--", label=f"duty {d:g}: m1 of U")
        a.plot(vs, [amp(x, q) for x in tr], c=c, ls="-.", label=f"duty {d:g}: r swing (abs. fundamental)")
        a.plot(vs, info / info.max(), c=c, ls=":", lw=2.5,
               label=f"duty {d:g}: delay information / max (best V/V$_\\pi$ = {best:.2f})")
        a.axvline(best, c=c, lw=0.8)
        print(f"  duty {d:g}: delay information maximal at V/V_pi = {best:.2f}, r swing maximal at "
              f"{vs[np.argmax([amp(x, q) for x in tr])]:.2f}")
    a.set_xlabel("V / V$_\\pi$")
    a.set_ylim(0, 1.3)
    a.set_title(f"G and U modulation depth vs the photon-limited optimum ({tag})", fontsize=9)
    a.legend(fontsize=7)
    for x in ax.flat:
        x.grid(alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=80)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
