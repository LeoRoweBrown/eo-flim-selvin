# EO-FLIM method notes

Working reference for analysing EO-FLIM data from this setup: the method, this repo's conventions, how to run the pipeline,
and pitfalls found in real datasets. Add to the dataset log at the bottom for each new day of data.

> Literature: the analysis code follows Bowman, Klopfer, Juffmann & Kasevich, "Electro-optic imaging enables efficient
> wide-field fluorescence lifetime microscopy", *Nat. Commun.* 10, 4561 (2019) (ratio method, Fig. 3c-style
> per-molecule error analysis). The papers referred to on 2026-10-31 were not received; summarise them here when added.

## 1. Measurement principle

- A Pockels cell driven at RF frequency f_RF rotates the emission polarisation. A polarising beamsplitter sends the light to two
  views on one camera: **U** (top half, "ungated") and **G** (bottom half, "gated", warped onto U by the bead registration).
  U + G collects all the light, so per frame r = G/(G+U) is independent of excitation drift and blinking.
- Gate: g(t) = sin²(Γ/2 · sin(ω_RF t − φ) + δ/2), with Γ = π V/V_π and δ the static retardance bias. With δ = 0, g repeats
  every half RF period: the **gating frequency is 2 f_RF = 40 MHz** (period 180° of RF phase).
- Excitation is a **50 % duty square wave**, not a pulse. The zero-lifetime trace is the gate convolved with the excitation
  (`convolve_g_with_square.py`). For a 50 % duty cycle the even gating harmonics cancel and the odd ones scale by 2/(πn).
- Phase stepping: the DDS steps the RF phase between frames (default 6 × 30° RF = 0–150°, i.e. 6 × 60° of gating phase over
  one 180° gating period). **The DDS steps positively, so the waveform moves backwards: `--phase-sign -1`.**
- With any excitation waveform, per harmonic n: C_n(sample) = C_n(IRF) · H_n(τ)/H_n(τ_ref), with H_n = 1/(1 + i n Ω τ) and
  Ω = 2π · 40 MHz = 0.2513 rad/ns. The IRF is measured on methylene blue (MB, 100 µM).

## 2. Lifetime estimators (all relative to the IRF at the same pixel)

| estimator | what it uses | notes |
|---|---|---|
| r_phase | arg of the 1st harmonic of r_k | robust to drift and to unmodulated light; **the most trustworthy here** |
| r_mod | modulus of the same | biased long by any unmodulated light |
| U/G phase/mod | single-view phasors | **biased by bleaching or drift within the 6 frames** (U and G shift in opposite directions) |
| ratio (LUT) | weighted fit of r_k to R_k(τ) curves | assumes a single exponential, so it sits between phase and mod when the sample is not one |

- τ_phase = −sign · tan(arg h)/Ω; τ_mod = sqrt(1/|h|² − 1)/Ω.
- Unmodulated fraction at fixed phase: f_dc = 1 − |h|/cos(arg h) (`bulk_phasor.py`). This is the share of light that must be
  unmodulated, or much longer-lived than 1/Ω, to explain a point inside the semicircle.
- τ_ref: the LUTs so far use τ_ref = 0 (MB treated as zero-lifetime). MB's real lifetime is a few hundred ps (check a reference
  value). At 40 MHz, τ_ref = 0.37 ns adds 5.3° of phase: a measured 0.61 ns becomes ≈1.0 ns, and 1.0 ns becomes ≈1.4 ns.
  Always state which τ_ref was used.

## 3. Pipeline (scripts in repo root)

Python: `C:\Users\Leo\venvs\image\Scripts\python.exe` (a pip venv; the conda env `image` named in the docstrings doesn't exist
on this PC, and miniforge's libmamba solver is broken). Raw files are 6-frame `(6, 3200, 3200)` uint16 ImageJ tifs, one per
acquisition. Pass them with `register_views.py ... --files-in-root`.

```bash
PY=/c/Users/Leo/venvs/image/Scripts/python.exe
D=<day folder>; A=$D/analysis_claude
$PY bead_registration.py --stack "$D/beads/<brightest>.tif" --out "$A/bead_registration"
$PY register_views.py "$D/<sub>"/*.tif --files-in-root --reg "$A/bead_registration/bead_registration.json" --out "$A/registered/<sub>"
MPLBACKEND=Agg $PY modulation.py "$A/registered/modulation" --out "$A/modulation"
$PY lifetime_lut.py "$A/registered/modulation/<irf>_registered.tif" --phase-sign -1 --out "$A/lut"
$PY bulk_phasor.py --irf "$A/registered/modulation" --sample "$A/registered/cy5" --blank "$A/registered/blank" --out "$A/bulk_phasor"
$PY lifetime_estimators.py --irf <irf_registered.tif> --lut <irf_lut.npz> --bulk <samples...> --out "$A/estimators/<irf>"
# single molecules / beads: sm_lifetime.py, superpixel_lifetime.py (see their docstrings)
```

Beads: `bulk_phasor.py --sample <beads> --bin 2 --spots` gives per-bead phasors (with the local background analysed as
`<name>_localbg`). `sm_lifetime.py <registered bead tif> --lut <power-matched lut> --reg <bead_registration.json>`
runs the single-molecule pipeline.

`bulk_phasor.py` writes a standard figure per acquisition (`acquisitions/`), plus an IRF-vs-IRF matrix, IRF drift vs time/power, phasor plots, power/exposure/time trends,
per-pixel |h| and τ vs intensity, blank-subtracted phasors, and a CSV with one row per stack.

## 4. Pitfalls found in real data (check each new dataset)

1. **The IRF depends on laser power: always use power-matched IRFs.** The MB raw phase moves about 14° of gating phase
   (≈1 ns) from 2 to 30 mW, and m1 drops from 0.47 to 0.40. Beads, which respond linearly to power, shift the same way, so
   this is the laser's excitation waveform, not the dye. The down-power series reproduces the up series (no hysteresis), and
   repeated same-power IRFs agree within ~0.05 ns over ~1 h. `bulk_phasor.py` defaults to `--match power`.
2. **Phase-stepping desync**: some stacks come out shifted by whole frames (2 frames = 120°, 3 frames = 180°). On 2026-10-30
   that was IRF `180751` and beads `173319` and `173614`. `bulk_phasor.py` excludes IRFs off their same-power peers by
   >30° and flags samples whose phase relative to the IRF is outside −20…90°.
3. **Bleaching within the acquisition** at long exposures (100–200 ms): U+G falls 3–9 % over the 6 frames. r-based estimators
   are immune; U/G single-view estimators are not (U vs G τ_phase differed by up to 0.6 ns).
4. **Bulk dyes saturate or bleach**. Signal per mW·ms falls about 20× (Cy5) and about 4× (MB) from 2 to 30 mW, while beads
   stay linear. Power labels are not proportional to photons.
5. U+G varies about ±2 % with phase even for MB (unequal channel efficiencies or phase-linked excitation). Frame
   normalisation and r remove it.
6. The auto-detected split row (1574 on 2026-10-30) can differ from the lab PC JSON (1586). The affine absorbs the
   difference, but always register with a JSON computed from that day's beads.

## 5. Bulk-dye demodulation (open problem)

Diagnostic plan: [demodulation_diagnosis_plan.md](demodulation_diagnosis_plan.md). It includes the 2026-11-05 update
(f_dc depends on exposure; oxygen scavengers don't remove it) and how a prompt component mimics demodulation, with a
two-frequency (20/40 MHz) unmixing protocol.

Remember that f_dc assumes "one lifetime + DC". Any mixture of lifetimes lies inside the semicircle. A prompt component
gives f_dc ≤ 0.14 with a 3.5 ns dye and ≤ 0.015 with a 1 ns dye.

Symptom: bulk Cy5 is heavily demodulated relative to MB (τ_mod 2–7 ns ≫ τ_phase), and worse at higher power.

Evidence from 2026-10-30:
- The blank dish background is ≤1.4 % of the Cy5 signal. Subtracting it changes τ by ≤0.01 ns, so **the blank does not
  explain it**.
- f_dc ≈ 0.15–0.35 at TIRF positions 4645/4647, and ≈ 0.35–0.47 at 2154/2158. It rises with power within a series.
- **At 20 MHz excitation f_dc is the same (0.26–0.46)**. A finite-lifetime component would demodulate less at 20 MHz, so
  the extra light is effectively unmodulated (or ≫ 10 ns).
- Within one image, Cy5 τ_phase rises roughly linearly with the local emission rate (≈0.1 ns in dim pixels to ≈0.75 ns in
  bright ones). Pixel phasors lie on streaks from near (1, 0) towards an interior point at about 1.1 ns phase and |h| ≈ 0.5.

Leading hypothesis (untested): **the MB IRF only calibrates light with MB's pupil filling**. The Pockels cell sits in a pupil
plane, so the ray angle through the crystal depends only on the camera pixel, whatever the depth. That is why the gate barely
changes with z, and why an angle-dependent retardance can't be the cause. What does depend on the source geometry is
*where* the light crosses the pupil, and **the modulation varies strongly across the pupil** (user, 2026-10-31).
- A defocused point source sends each pixel the light from one part of the pupil, while MB fills the whole pupil at every
  pixel.
- Out-of-focus light from an inclined beam or a limited excited region fills the pupil unevenly.
- Surface emitters also emit supercritical-angle fluorescence into the pupil rim, which bulk emitters don't.

1 µM Cy5 is excited along mm of beam path; 100 µM MB absorbs within ~0.5 mm. So Cy5 (and the glow around beads) is
weighted towards a different part of the pupil.

- A 20 vs 40 MHz comparison tells *which kind* of pupil variation matters. The gate's harmonics are ∝ J₂(Γ)cos δ at
  40 MHz and J₁(Γ)sin δ at 20 MHz. If the drive (Γ, δ) varies across the pupil, the two frequencies are demodulated
  differently. If instead the fraction of light that is gated at all (extinction, depolarisation, leakage) varies, both
  are demodulated by the same factor.
- At matched TIRF positions, f_dc is similar at both frequencies:

  | TIRF position | 40 MHz | 20 MHz |
  |---|---|---|
  | 2154/2158 | 0.35–0.47 | 0.38–0.43 |
  | 4645/4647, 2 mW | 0.14–0.24 | 0.27–0.30 |

  This weakly favours the "fraction gated" kind.
- **Counter-evidence (morning of 2026-10-30): 1 µM MB is not demodulated while 1 µM Cy5 is**, at the same concentration
  and TIRF setting. If this holds with a properly bracketed IRF, the cause is specific to Cy5 (photophysics, a long-lived
  impurity or photoproduct, or an unmodulated excitation component that Cy5 absorbs more strongly) rather than geometry.
  Test: Cy5 at 100 µM, and a non-isomerising red dye (e.g. ATTO 655) at 1 µM.

Tests:
1. Measure 1 µM MB (no inner filter) against 100 µM MB. If it looks "demodulated", the hypothesis holds.
2. Use a thin Cy5 layer (a film between coverslips, or a dense surface-immobilised layer), or a high Cy5 concentration.
3. Add a field or aperture stop at an intermediate image plane to cut out-of-focus light, then repeat bulk Cy5.
4. Use a zero-lifetime scatter reference (a thin scattering film with the emission filter removed or ND'd) as a second IRF.

Alternative to keep in mind: an unmodulated spectral component of the excitation (for example a diode's broadband spontaneous
emission in the "off" half-cycle) that MB and Cy5 absorb differently. A laser-line clean-up filter would test this.

## 6. 20 MHz excitation (50 % duty) with 40 MHz-capable gating

- A 50 % square wave at 20 MHz has only odd harmonics of 20 MHz, and the unbiased gate only even ones, so there would be no
  signal. Gating at 20 MHz comes from the bias δ: the gate gains a J₁(Γ) sin δ component at 20 MHz (this is why U ≠ G).
- The trace then repeats every **360° RF**, and 6 × 30° steps only cover 150°. Fit DC plus the fundamental with period 360°:
  `bulk_phasor.py --f-gate-mhz 20 --period 360 --harmonics 1`. A 3rd harmonic of ~10 % is present but can't be separated
  from 6 points; it moves τ_phase by up to ~0.5 ns.
- Better acquisition: 6–8 steps covering the full 360° (e.g. 60° or 45° steps), so the 1st and 3rd harmonics can both be fitted.

## 7. Dataset log

### 2026-11-05 `F:\uiuc_data\20261105(1)\20261105\ox_test` (analysed 2026-11-05 → `analysis_claude/`)
- DIY channel slide, PCA/PCD + Trolox. Channels: ATTO–streptavidin and Cy5, each ~1.2 µM, unpassivated glass. All
  at 10 mW, TIRF 3773, attenuated with OD 0/0.5. z is the focus offset from the surface as named.
- **New registration**: beads `bead_stack_pockels_off_lc1,5vrms_1/2/3` agree within 0.1 px (CV 0.52 px). Split 1522
  (was 1574), x offset −39 px (was +2), so don't reuse the 2026-10-30 JSON.
- **MB IRFs are very stable**: raw phase −86.4 → −86.6° from 17:53 to 18:44 (0.017 ns), and |h| 0.46–0.48 at
  10–40 ms. The 200 ms MB `184251` saturates (65535 ADU) and is excluded.
- The camera dark level rises with exposure (~100 + 0.2 ADU/ms, everywhere on the chip). The margin offset follows it.
- Analysis: `z_time/` (scratchpad `ox_z_time.py`). Phasors are per-pixel calibrated, intensity-weighted, against an IRF
  interpolated in time between the two MB brackets. They agree with the ROI ratio of sums within 0.01.
  `bulk_phasor/` has the standard per-acquisition figures.
- **Still demodulated with oxygen scavengers**: Cy5 f_dc 0.23–0.35, τ_phase 1.24–1.30 ns. ATTO f_dc 0.12–0.27,
  τ_phase 1.68–1.93 ns (early, before clumping).
- **f_dc rises with exposure, reversibly, at fixed power and field; τ_phase is unchanged.**
  - ATTO, one field: 20 ms 0.12 → 40 ms 0.17 → 100 ms 0.26 → 200 ms 0.27, then 40 ms 0.17 and 20 ms 0.13 again.
  - Cy5: 100 ms 0.23 → 200 ms 0.31 → 400 ms 0.35, then 100 ms 0.23 again.
  - MB IRFs show no trend from 10 to 40 ms.
  - Stray/dark light is far too small to explain it (≤40 ADU at 200 ms, against the ~400 ADU of extra unmodulated
    signal needed).
  - Pupil/defocus geometry cannot depend on exposure. Untested: unsaturated MB at 100–400 ms, and exposure at constant
    counts (OD).
- **No z dependence**: Cy5 −2…+10 µm gives f_dc 0.33 → 0.30. ATTO 200 ms 15 → 0 µm gives 0.29–0.34. But for a uniform
  layer with the cell in the pupil, defocus does not change the pupil distribution per pixel, so this does not test
  H1a (that needs point sources).
- **Time**: ATTO τ_phase rose 1.68 → 1.93 ns between 17:58 and 18:07, while f_dc at fixed exposure did not change.
  By 18:24 the ATTO signal had fallen ~25–80× (clumping/sticking). After that, τ_phase was 2.5–2.8 ns and f_dc
  0.29–0.35 at 200 ms. The 40 ms z-series (`182719`–`183015`, 5–6 ADU/px) is unusable. Cy5 was stable, with a
  field-to-field f_dc difference (0.23 vs 0.32 at 100 ms).
- `trig`/`notrig` and the 0.5 s laser settle made no difference. The first Cy5 stack `180820` (OD 0.5, f_dc 0.65) is
  anomalous.

### 2026-10-30 (morning) `F:\uiuc_data\20261030_cy5_bulktest\20261030_cy5_bulktest` (analysed 2026-11-01 → `analysis_claude/`)
- RF 17.15 W, 2.5 Vrms, TIRF 3970. No beads were taken, so registration used the bead JSON from the afternoon session.
- **The gate changed during the session.** Raw IRF |h(r)| fell from 0.46–0.49 (100 µM MB, 11:47) to 0.31–0.33
  (100 µM MB, 12:55, 1–15 mW). The raw phase at matched power was unchanged (81° at 10 mW in both series). A dim MB stack
  at 12:30 (`123012`) gave 0.21. **Use IRFs bracketing each sample in time, not just matched in power.**
- **1 µM MB is not demodulated**: against the 12:55 IRF series, its f_dc is 0.00 ± 0.02. Its raw |h| is 0.32, the same as
  the IRF. τ_phase ≈ −0.4 ns, which is only the IRF power trend (IRF 15 mW vs sample 20–50 mW).
  - Caveat: the 1 µM MB was taken 15 min before that IRF series, with no IRF in between.
- **1 µM Cy5 is demodulated against the same IRFs**: f_dc 0.33–0.45. τ_phase is 1.07–1.32 ns where the IRF power matches.
  - This holds both against the 11:47 series (early Cy5) and the 12:55 series (late Cy5).
- 1 µM MB and 1 µM Cy5 have similar bulk geometry, so this points to **something specific to Cy5**, not to out-of-focus or
  pupil geometry (see section 5).
- Desynced: Cy5 `115633` (+60°, one frame; not caught by the −20…90° flag) and `122108`.
- Blank `blankno1`: ≤6 % of the Cy5 signal at 40 mW.

### 2026-10-30 `F:\uiuc_data\20261030_cy5_bulktest_2\20261030_cy5_bulktest_2` (analysed 2026-10-31 → `analysis_claude/`)
- RF 17.03 W, 2.7 Vrms (40 MHz gating); 1.3 Vrms for the 20 MHz runs. Registration from beads `173819`: 373 beads,
  CV 0.62 px, split 1574.
- MB IRFs: 4645 (before Cy5), 4645 after Cy5, 2152, 4647, and 4647 stepped down in power. IRF `180751` is desynced.
- Cy5 1 µM against power-matched IRFs (τ_ref = 0), ROI r phasor:
  - **4645:** τ_phase 0.29–1.05 ns, rising with power and local intensity; τ_mod 2.1–4.8 ns; f_dc 0.09–0.34.
  - **2154:** τ_phase 1.06–1.26 ns, τ_mod 5–6.6 ns, f_dc 0.35–0.47.
  - **20 MHz:** τ_phase 0.8–1.4 ns, f_dc 0.27–0.43.
- Blank dish: ≤1.4 % of the Cy5 signal (2154); ~0 at 4346.
- Beads (2154, 10–40 mW, power-matched IRFs, `bulk_phasor.py --bin 2 --spots`; desynced `173319` and `173614`
  excluded): per-bead phasor of the brighter half gives τ_phase 3.5–4.0 ns, τ_mod 3.9–4.9 ns, f_dc ≈ 0
  (single-exponential), with no power trend. `sm_lifetime.py` (ratio LUT) gives medians of 3.43–3.74 ns
  (IQR 1.3–1.85), residual G−U shift ≤0.13 px, χ²/dof ≈ 2.
  **The diffuse glow around the beads is ~77 % unmodulated** (τ_mod 26–33 ns), the same signature as bulk Cy5. In-focus
  emitters measure correctly and diffuse light is demodulated, which supports the hypothesis in section 5.
