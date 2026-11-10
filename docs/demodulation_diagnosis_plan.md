# Diagnosing bulk-dye demodulation: experimental plan

Draft, 2026-11-01. The background, evidence and analysis conventions are in
[eo_flim_method_notes.md](eo_flim_method_notes.md), section 5 and the dataset log.

## The problem in one line

Relative to the methylene blue (MB) IRF, the phase-stepped ratio r = G/(G+U) swings less for bulk dyes (Cy5, ATTO) than
for MB or in-focus beads. For example, on 2026-10-30 at 10 mW, the peak-to-peak of r was 0.30 for MB and 0.20 for Cy5. In
phasor terms, 15–60 % of the dye light behaves as if it were not gated at 40 MHz.

## What is already established

- **The analysis maths is right.** Synthetic single exponentials (1 and 3.5 ns) pushed through `bulk_phasor.py` come back
  exact. This holds with a G/U efficiency imbalance of ±15 %, a static bias of 20° and V/Vπ = 0.8. The demodulation is also
  visible in the raw per-step r, with no phasor maths at all (`raw_r_steps.png`).
- **Only one thing in the data reproduces it**: extra light that adds equally to U and G, i.e. light that isn't gated or
  doesn't vary at 40 MHz. A 30 % additive share gives f_dc ≈ 0.31. A phase-step calibration error gives a *negative* f_dc,
  and an efficiency imbalance gives almost nothing.
- **Ruled out or unlikely:**
  - The blank (water and dish): ≤1.4–6 % of the signal.
  - Exposure-time effects in the reference: beads at 100 ms and 1 µM MB at 20–50 ms are fine. *But the dyes' f_dc
    does depend on exposure (2026-11-05, see "Update 2026-11-05" below).*
  - A fault in the calculation (see above).
- **Fine (f_dc ≈ 0):**
  - In-focus beads on the coverslip.
  - 100 µM MB.
  - 1 µM MB (one session; the caveat is that its IRF came 15 min later).
- **Demodulated:**
  - Bulk 1 µM Cy5.
  - ATTO (always, according to the user).
  - The diffuse glow around the beads (77 %).

  For Cy5:
  - f_dc rises with laser power.
  - f_dc depends on the TIRF position.
  - f_dc is about the same at 20 and 40 MHz.
  - The rings in the Cy5 image are modulated; the unmodulated light is spatially smooth.
- **Setup facts:**
  - The Pockels cell is in a pupil plane, and the user reports that the modulation varies strongly across the pupil.
  - The laser is modulated by switching the drive current directly.
  - The IRF depends on power and drifts within a session.

## Hypotheses

| | hypothesis | explains | strains to explain |
|---|---|---|---|
| **H1** | **Surface vs bulk origin.** Light from emitters at the coverslip is gated like the IRF; light from the bulk is not. MB is a cationic dye that adsorbs to glass, so even 1 µM MB may be dominated by an adsorbed surface layer. | beads fine and their glow not; ATTO and Cy5 both bad; dependence on TIRF position; f_dc rising with power (the surface layer bleaches, the bulk is replenished by diffusion); smooth unmodulated light | nothing yet |
| H1a | ...because the gate varies across the pupil, and surface emitters fill the pupil differently from bulk emitters. Emitters within about λ of the glass emit strongly into angles beyond the critical angle (NA 1.33–1.49, the pupil rim; supercritical-angle fluorescence); bulk emitters don't. | the strong variation across the pupil | a phase shift should come with it |
| H1b | ...because some bulk or out-of-focus light reaches the camera without being gated (scatter off the crystal walls or electrodes, depolarisation, leakage around the polariser). | f_dc being frequency-independent | |
| **H2** | **Part of the excitation is unmodulated**: the diode's spontaneous emission or amplified spontaneous emission (ASE). Above threshold the carrier density clamps, so this light is nearly constant, broadband and incoherent (no rings). Dyes absorb it differently from the laser line. | smooth unmodulated light; dye dependence | beads fine (their dye would absorb it too) |
| H3 | **Dye photophysics or chemistry**: a long-lived (≫25 ns) emitting photoproduct, impurity or delayed emission. | f_dc rising with power | ATTO and Cy5 both affected; the bead glow |
| H4 | **The measurement itself**: gate drift between IRF and sample, IRF power mismatch, phase-step desync, camera offset. | the size of some scatter | the effect is systematic and dye-dependent |
| H5 | **A prompt (≈0 ns) component** mixed with the dye: leaked excitation (the TIRF reflection is strong), Raman, or strongly quenched dye. It lies on the chord to (1, 0), not on the line to the origin (see below). | ATTO's short τ_phase (1.7–1.9 ns, against 3.5 ns for ATTO 647N) | at most f_dc ≈ 0.14 with a 3.5 ns dye and ≈ 0.015 with Cy5, so it cannot be the whole effect |

H1 is the leading hypothesis. The plan tests it first and most directly, while running the cheap checks for H2 and H4 in
parallel.

## Update 2026-11-05 (oxygen scavengers, ATTO 647N–streptavidin and Cy5)

Data: `F:\uiuc_data\20261105(1)\20261105\ox_test`; details in the method notes' dataset log.

- **Still demodulated with PCA/PCD + Trolox:** Cy5 f_dc 0.23–0.35 (0.33–0.45 without scavengers on 2026-10-30), ATTO
  0.12–0.27. Removing oxygen lengthens triplet lifetimes, so delayed emission from triplets would have *grown*. It
  didn't, which weakens a triplet-based H3.
- **f_dc rises with exposure, reversibly, at fixed power and field; τ_phase is unchanged:**
  - ATTO: 20 ms 0.12 → 40 ms 0.17 → 100 ms 0.26 → 200 ms 0.27, then back to 0.17 and 0.13.
  - Cy5: 100 ms 0.23 → 200 ms 0.31 → 400 ms 0.35, then back to 0.23.
  - MB IRFs show no trend from 10 to 40 ms. The 200 ms MB saturated, so long-exposure IRFs are untested.
  - Stray/dark light (~0.2 ADU/ms) is ~10× too small to explain it.
  - Geometry (H1) cannot depend on exposure.
- **Laser:** off between acquisitions and, per the user, on during them. The total U+G changes by <2 % across the 6
  frames at every exposure, and the 0.5 s laser-settle stacks made no difference, so nothing visibly builds up during a
  stack. *Steady-state photophysics under continuous illumination should not depend on exposure, so the mechanism of
  the exposure dependence is still unknown.* Open: what `trig`/`notrig` switch, and whether the laser is gated per frame.
- **No z dependence** (Cy5 −2…+10 µm, ATTO 0…15 µm). For a uniform dye layer with the cell in the pupil, defocus does
  not change the pupil distribution reaching a pixel, so this does not test H1a; that needs point sources (E1).
- **ATTO τ_phase is short and changes:** 1.68–1.93 ns before clumping and 2.5–2.8 ns after (2.1–2.4 and 3.2–3.3 ns with
  τ_ref = 0.37 ns for MB), against ~3.5 ns for ATTO 647N. The user's earlier ATTO measurements gave 2.77 ns.

## Prompt light, long-lived light and two-frequency unmixing

### What "demodulated" means in the phasor

f_dc does not measure a loss of modulation as such. It measures how far the point lies *inside* the semicircle for its
phase:

    f_dc = 1 − |h| / cos(arg h)

A single exponential lies on the semicircle (f_dc = 0). **Any mixture of different lifetimes lies inside it**, because
the phasor of a mixture is the intensity-weighted average of the components' phasors, i.e. a point on the chord between
them. The "τ + DC" reading is only one of the mixtures that reach a given point:

| mixture | where it lies | what changes |
|---|---|---|
| τ + unmodulated (DC, or lifetime ≫ 1/ω) | on the line from the τ point to the origin (0, 0) | |h| falls, phase unchanged; f_dc = the DC fraction at every frequency |
| τ + prompt (≈0 ns) | on the chord from the τ point to (1, 0) | phase falls (τ_phase shortens), |h| *rises*, but by less than a single exponential at that phase would need |
| τ + a longer component (5–100 ns) | on the chord to that component's point | between the two; approaches the DC case as the long lifetime grows |

### How prompt light makes a point look demodulated

Prompt light is fully modulated and has zero phase relative to the IRF, so on its own it would *raise* the modulation.
It looks demodulated because it lowers the phase *more* than it raises |h|. The point then sits below the semicircle at
its new phase, and the "τ + DC" reading interprets that gap as unmodulated light. Example: 80 % of 3.5 ns plus 20 %
prompt, at 40 MHz:

- h = 0.8 · (0.564 + 0.496i) + 0.2 = 0.651 + 0.397i, |h| = 0.762, phase 31.4°;
- τ_phase = 2.42 ns, τ_mod = 3.38 ns, f_dc = 0.11, with no unmodulated light at all.

Apparent f_dc from a prompt component (computed, 1 − long fraction = prompt fraction):

| long component | prompt fraction | f_dc at 20 MHz | 40 MHz | 80 MHz | τ_phase at 40 MHz |
|---|---|---|---|---|---|
| 3.5 ns | 10 % | 0.017 | 0.065 | 0.21 | 2.92 ns |
| 3.5 ns | 20 % | 0.030 | 0.107 | 0.31 | 2.42 ns |
| 3.5 ns | 30 % | 0.038 | 0.132 | 0.34 | 1.99 ns |
| 3.5 ns | 50 % | 0.044 | 0.139 | 0.30 | 1.26 ns |
| 1.0 ns (Cy5-like) | any | ≤0.004 | ≤0.015 | ≤0.056 | |

Consequences:
- **Prompt light cannot explain Cy5's f_dc of 0.23–0.35.** With a 1 ns dye the chord to (1, 0) hugs the semicircle. Cy5
  needs truly unmodulated light or a component much longer than 1/ω.
- **For ATTO it can explain part of it**: early ATTO at 20 ms (τ_ref-corrected (0.70, 0.40); 40 ms (0.65, 0.38)) is
  close to 3.5 ns plus ~20 % prompt light. The rise of f_dc with exposure on top of that moves the point towards the
  origin, i.e. it is DC-like.
- Clumping lengthened ATTO's τ_phase while the total signal fell 25–80×. A fixed prompt background (leakage, Raman)
  would have become a *larger* share, so the short component probably came from the dye in solution itself (quenched or
  stacked dyes on multiply labelled streptavidin, or aggregates that later stuck to the glass). This is an inference.

### Unmixing with two frequencies

One frequency gives two numbers (g, s) per pixel, so it cannot tell the mixtures above apart. **A second frequency on
the same field adds two more**, and the mixtures have different frequency signatures:

| what is mixed with the dye | f_dc at 20 MHz vs 40 MHz |
|---|---|
| truly unmodulated light (DC, or ≫ 100 ns) | the same at both |
| a 10–30 ns component (e.g. 30 % of 10 ns with Cy5) | lower at 20 MHz (0.13 vs 0.20 for 10 ns; nearly equal at 30 ns) |
| a prompt component | ~3–4× lower at 20 MHz (0.03 vs 0.11 for 20 % with 3.5 ns) |

With the dye lifetime known (ATTO 647N 3.5 ns, Cy5 ~1 ns), the model "dye + prompt + DC" is linear in the three
fractions (they sum to 1). The two complex phasors give four equations for two free fractions, so the fit is
overdetermined and its residual tests the model. If the residual is large, let the second lifetime float.

The 2026-10-30 data already gave f_dc ≈ the same at 20 and 40 MHz for Cy5 (0.27–0.43 vs 0.35–0.47), which favours
DC-like light for Cy5. That comparison was not on the same field or at the same time, so redo it properly:

1. Same field, same exposure and power, interleaved 40 → 20 → 40 MHz, with an MB IRF at each frequency before and after.
2. Calibrate per pixel at both frequencies. The 20 MHz IRF varies strongly across the field (|h| 0.65–1.03, phase
   −2.8…+1.4 ns), so ROI averages are unsafe there.
3. 20 MHz has a ~10 % 3rd harmonic that 6 steps cannot separate (up to ~0.5 ns on τ_phase). Use 12 steps × 30° (a full
   360° RF period) at 20 MHz so it can be fitted.
4. Samples: ATTO 647N (free dye and the streptavidin conjugate), Cy5, a buffer-only channel (any prompt light from
   leakage or Raman appears there at ≈0 phase), and 1 µM MB.
5. Fit dye + prompt + DC per pixel with the dye lifetime fixed, and report the fractions and residuals.

**A second frequency without switching modes:** a 50 % duty square wave at 40 MHz has no 80 MHz harmonic, so the current
data carry no 80 MHz information. A ~25–35 % duty cycle adds one. The gate's 80 MHz term goes as J₄(Γ) (0.15 at V = Vπ,
against J₂ = 0.49 at 40 MHz), so it would be weak but simultaneous. The table above shows that 80 MHz is the most
sensitive to a prompt component. This needs 8+ phase steps.

### Which harmonics each configuration gives (combining 20/40/80 MHz)

The phase-stepped trace r(θ) over RF phase θ is the gate convolved with the excitation-convolved decay. Its Fourier
component at RF harmonic m (frequency m × 20 MHz) is

    R_m = gate_m · exc_m · D(ω_m),   D(ω) = Σ_k a_k / (1 + iωτ_k)

So **a frequency is usable only where both the gate and the excitation have a component**:
- **Gate:** even m ∝ J_m(Γ) cos δ, odd m ∝ J_m(Γ) sin δ.
- **Square-wave excitation at f_exc with duty d:** harmonics at n · f_exc with amplitude ∝ sinc(n d). At 50 % duty only
  odd n survive; at 25 % duty n = 4, 8, … vanish.

Amplitude of the r swing and photon information (swing² / mean r(1 − r), ∝ 1/variance), from the
`convolve_g_with_square.py` model:

| configuration | 20 MHz | 40 MHz | 60 MHz | 80 MHz | steps needed |
|---|---|---|---|---|---|
| 40 MHz exc, 50 %, V/Vπ 1.0, δ 0 (current) | – | 0.31 (info 0.53) | – | – | 6 × 30° |
| 40 MHz exc, 25 %, V/Vπ 1.0, δ 0 | – | 0.44 (1.51) | – | 0.10 (0.07) | 8 × 22.5° over 180° |
| 40 MHz exc, 25 %, V/Vπ 1.3, δ 0 | – | 0.31 (0.65) | – | 0.19 (0.24) | 8 × 22.5° over 180° |
| 20 MHz exc, 50 %, V/Vπ 0.6, δ 90° | 0.37 (0.76) | – | 0.02 | – | 6–12 × 30° |
| 20 MHz exc, 25 %, V/Vπ 0.6, δ 90° | 0.52 (2.44) | – | 0.03 | – | 12 × 30° |
| **20 MHz exc, 25 %, V/Vπ 0.8, δ 45°** | **0.31 (0.55)** | **0.20 (0.23)** | 0.05 (0.01) | 0 | **12 × 30° over 360°** |
| 20 MHz exc, 25 %, V/Vπ 1.0, δ 45° | 0.18 (0.17) | 0.22 (0.24) | 0.07 (0.03) | 0 | 12 × 30° over 360° |

Notes:
- **The 25 % duty cycle at 40 MHz alone roughly triples the 40 MHz information** (1.51 vs 0.53) and adds a weak 80 MHz
  term. V/Vπ ≈ 1.3 strengthens 80 MHz at the cost of 40 MHz.
- **20 MHz excitation at 25 % duty with δ ≈ 45° gives 20 and 40 MHz in the same frames.** There is then no drift,
  exposure or sample change between the frequencies, and the pixels are identical. This is the cleanest unmixing
  measurement. δ = 90° kills the even harmonics (no 40 MHz); δ = 0 kills the odd ones (no 20 MHz).
- **Steps:** N steps over the trace period resolve harmonics up to (N − 1)/2 without aliasing. The 40 MHz-excitation
  trace repeats every 180° of RF, so 8 × 22.5° resolves 40, 80 and 120 MHz. The 20 MHz-excitation trace repeats every
  360°, so 12 × 30° resolves 20–100 MHz. Use `--period` and `--harmonics` in `bulk_phasor.py` to match.
- These are model numbers for an ideal square wave. At 25 % duty the 6 ns (40 MHz) or 12 ns (20 MHz) pulses are
  distorted by the diode's turn-on delay. The IRF absorbs this only if it is taken at identical settings (duty, δ, V,
  power, exposure).

### Unmixing recipe

1. **Per harmonic, per pixel, relative to an IRF with identical settings:**
   D(ω_m) = [h_m(sample) / h_m(IRF)] · L_ref(ω_m), with L_ref(ω) = 1 / (1 + iωτ_ref).
   - τ_ref matters more at higher frequency (MB at 0.37 ns is 5.3° at 40 MHz, 10.5° at 80 MHz).
   - Use a scattering solution as the IRF (τ_ref = 0 exactly), or measure the MB lifetime against one.
2. **Stack every usable D(ω_m) from every acquisition of the same field**: harmonics of one run, and runs at other
   configurations.
   - Runs must use the same exposure and counts, because f_dc depends on exposure (2026-11-05). Interleave them
     (A → B → A) to check that the sample didn't change.
3. **Fit a mixture with intensity fractions:**
   D(ω) = a_dye L(τ_dye, ω) + a_prompt · 1 + a_long L(τ_long, ω), with a_DC = 1 − a_dye − a_prompt − a_long.
   - DC light contributes nothing at any ω > 0 and enters only through the normalisation by c₀. That is why it has
     no column.
   - Given the lifetimes, the fit is linear in the fractions: bounded least squares (`scipy.optimize.lsq_linear`,
     0 ≤ a ≤ 1, plus a penalty row for Σa ≤ 1).
   - Fit τ_long, or τ_dye, by a 1-D grid search around the linear fit.
   - Weight each D(ω_m) by its information (the table above, or measured from the pixel-to-pixel scatter of D over a
     uniform patch).
4. **Count degrees of freedom.** Each frequency gives 2 real equations.

   | frequencies | equations | fixed τ_dye; dye + prompt + DC (2 free) | free τ_long; + long (4 free) |
   |---|---|---|---|
   | 2 | 4 | 2 spare (the residual tests the model) | 0 spare |
   | 3 | 6 | 4 spare | 2 spare |

   So **to test "dye + prompt + DC + a free long component", three usable frequencies are needed.** Options are 20 + 40
   (one run, δ 45°) plus a 40 + 80 run, or the 20/40/60 MHz set if the 60 MHz term is bright enough.
5. **Check the result on known samples:**
   - beads, which should come out as one lifetime;
   - a buffer-only channel (prompt plus background only);
   - free ATTO 647N (3.5 ns);
   - free Cy5 (~1.0 ns).
   Do this before trusting the fractions on the bulk dyes.

### Free Cy5: what the 2026-11-05 numbers say

- **Prompt light cannot explain Cy5.** With a 1 ns dye it gives at most f_dc ≈ 0.015, against the measured 0.23–0.35.
  The admixture is DC or long-lived.
- **Cy5's τ_phase is longer than ~1 ns**, which DC alone cannot do (DC doesn't move the phase). Two models fit the
  z-series field (100 ms; measured τ_phase 1.27, τ_mod 4.7 ns, f_dc 0.32, τ_ref = 0) about equally well:

  | model | τ_phase | τ_mod | f_dc |
  |---|---|---|---|
  | 1.0 ns Cy5 + 30 % of a ~30 ns component | 1.23 ns | 4.1 ns | 0.27 |
  | 1.0 ns Cy5 + 30 % of 10 ns | 1.53 ns | 3.5 ns | 0.20 |
  | Cy5 at ~1.3 ns (lengthened, e.g. adsorbed to glass, which hinders cis–trans isomerisation) + ~30 % DC | 1.27 ns | ≈4.7 ns | 0.32 |

- **τ_ref decides how plausible each is:**

  | τ_ref | Cy5 τ_phase | early ATTO τ_phase | implies |
  |---|---|---|---|
  | 0 | 1.24–1.30 ns | 1.68–1.93 ns | Cy5 plausible; ATTO needs ~30–40 % prompt light |
  | 0.2 ns | 1.46–1.49 ns | | |
  | 0.37 ns | 1.66–1.69 ns | 2.1–2.4 ns | Cy5 long for free dye; ATTO needs ~20 % prompt light |

  100 µM MB may be self-quenched below 0.37 ns. **Pin τ_ref down first** (MB against a scatter IRF).
- **Distinguishing the models:** a ~30 ns component gives almost the same f_dc at 20 and 40 MHz as pure DC (0.255 vs
  0.273). But it raises τ_phase more at 20 MHz, while DC leaves the phase alone. The fit in step 3 with τ_dye fixed at
  1.0 ns tells them apart.
- **Possible sources of a long or DC component** (none confirmed):
  - autofluorescence of the DIY channel slide (untested; the 2026-10-30 blank was a dish);
  - a buffer impurity (but Cy5 was demodulated without scavengers too);
  - unmodulated excitation (H2).
- **Tests:**
  - a buffer-only channel in the same slide (TIRF, exposure and filters unchanged);
  - Cy5 in glycerol, or Cy5-DNA, as a lengthened-Cy5 reference;
  - the two-frequency fit above.

### Other open candidates for the DC-like part

1. **Unmodulated excitation (H2)** from the directly modulated diode (spontaneous emission or a spectral pedestal). If
   it is at a different wavelength from the laser line, the dyes absorb it differently: Cy5 and ATTO 647N absorb at
   ~646–649 nm, MB at 664 nm. That would fit 1 µM MB being clean. Test with a laser-line clean-up filter (E4.2) and a
   log-scale spectrum with modulation on and off (E4.1).
2. **Exposure-dependent timing** (camera vs phase stepping vs laser). Test: MB and dye at 10 → 400 ms with OD chosen to
   keep the counts per frame constant. If MB also loses modulation at long exposure, it is instrumental.
3. **Millisecond luminescence:** switch the laser off part-way through a long exposure, or start an exposure just after
   switching it off. Any signal after switch-off is delayed emission.

## Rules for every session (controls H4)

1. **Bracket each sample block with IRFs.** Take 100 µM MB immediately before and after every block, at the same powers
   and exposures as the samples. The gate modulation fell from 0.48 to 0.32 within an hour on 2026-10-30.
2. **Fix the laser drive current and set the power at the sample optically** (half-wave plate and polariser, or ND
   filters). One IRF then serves all powers, and power is no longer confounded with the laser waveform. Record the
   current setting.
3. **Take a dark stack** (laser off, same exposure) for each exposure used. Compare it with the margin-based offset that
   `bulk_phasor.py` uses. The synthetic test shows that an offset error of 5 % of the signal gives f_dc ≈ 0.07.
4. **Repeat the first condition at the end of the session** to measure drift.
5. **Take beads every day for registration.** Use 8 phase steps × 22.5° (one full gating period) where acquisition time
   allows, so the 2nd and 3rd harmonics can be fitted and desyncs are easier to see.
6. **Check desyncs before trusting any stack.** The current flag window missed a one-frame (60°) shift.

## Experiments, in order

### E1. Defocused beads (most decisive for H1; ~1 h)

Sparse beads stuck to the coverslip, in water with no free dye.
1. Phase-step at z = 0, ±2, ±5, ±10, ±20, ±50 µm, with MB IRFs bracketing the series.
2. Analyse with `bulk_phasor.py --spots --bin 2`, which gives a per-bead phasor. At large defocus, use a fixed ROI around
   each bead instead.
3. Also map f_dc and phase across one strongly defocused bead disc. Each pixel of a defocused disc sees a different part
   of the pupil.

| result | interpretation |
|---|---|
| f_dc rises with defocus, and the disc map shows structure | **H1 confirmed with a single dye and lifetime.** Go to E3 for the mechanism. |
| a defocused bead stays at f_dc ≈ 0 | defocus alone isn't the cause. Then the difference between surface and bulk must come from the supercritical-angle light, or from being in solution. Go to E2. |

### E2. Surface vs bulk with the same dye (H1; ~half a day)

1. **Measure how deep the signal comes from.** For bulk 1 µM Cy5, 1 µM MB and 100 µM MB, phase-step with the focus at the
   surface and at 2, 10, 50 and 200 µm into the solution.
   - Under H1, f_dc rises as the focus moves away from the surface.
   - Under H1, f_dc for MB stays low at the surface but rises in the bulk if the adsorbed layer is what made MB look fine.
2. **Test MB adsorption.**
   1. Measure 1 µM MB.
   2. Rinse the dish 3× with water so only adsorbed MB remains, and measure again.
   3. Repeat on a passivated coverslip (PEG, BSA or Pluronic) to which MB can't stick.
   - Under H1, passivated 1 µM MB is demodulated like Cy5, and the rinsed dish still looks like the IRF.
3. **Measure dye bound to the surface with only buffer above it.** Use Cy5- or ATTO-labelled BSA or DNA, adsorbed or bound
   via biotin–streptavidin, dense enough to be bright. Then add 1 µM free dye on top and measure again.
   - Under H1: bound dye alone gives f_dc ≈ 0 and the correct lifetime; adding free dye makes it demodulated.
   - If bound dye is already demodulated, that points to H2 or H3.
4. **Vary the sample thickness.** Sandwich 1 µM Cy5 between a coverslip and a slide with spacers of about 5, 25 and
   100 µm, and compare with an open dish.
   - Under H1, f_dc rises with thickness.
5. **Measure a bulk zero-lifetime reference.** Use dilute scatterers (Ludox or polystyrene nanoparticles) suspended in
   water. Swap the emission filter for an ND filter plus a laser-line filter, so the scattered laser is imaged.
   - Under H1, scatter from the bulk is demodulated relative to MB, while scatter from particles stuck to the surface is not.
   - Note that scattered light is partly polarised; check that U+G stays constant with phase.

### E3. Measurements in the pupil plane (mechanism of H1, if E1 or E2 is positive; ~1 day)

1. **Image the pupil.** Use a Bertrand lens, or a camera at a pupil conjugate, with phase stepping. Image:
   - 100 µM MB, which gives the modulation and phase map across the pupil;
   - beads;
   - bulk 1 µM Cy5;
   - rinsed (adsorbed-only) MB.

   Then look for a bright ring beyond NA 1.33 (supercritical-angle light) in the surface samples and its absence in bulk
   Cy5.
2. **Close the loop with a prediction.** Compute h_pred = Σ w_sample(ρ) h(ρ) / Σ w_MB(ρ) h(ρ), where w(ρ) is each sample's
   intensity distribution over the pupil and h(ρ) is the MB modulation map. Compare h_pred with the measured bulk Cy5
   phasor. If they agree, the mechanism is understood quantitatively.
3. **Put stops in a pupil conjugate:**
   - an iris limiting to NA < 1.3, which cuts the supercritical ring;
   - an annulus that passes only the ring;
   - a small central iris.

   Repeat MB and bulk Cy5 with each. If an iris makes MB and Cy5 agree, it fixes the measurement (at some cost in light),
   or a bulk reference can be used as the IRF instead.
4. **Check whether light bypasses the gate (H1b).** Turn the RF off and use the LCVR to set a very unequal static split,
   r ≈ 0.1. Then measure r for MB, bulk Cy5 and the bead glow. The bypassing fraction is
   (r_sample − r_MB) / (0.5 − r_MB).
   - If this fraction matches f_dc, the cause is light that bypasses the gate.
   - If the fraction is 0 while f_dc is still high, the cause is a gate that varies across the pupil (H1a).
5. **Add a field stop** at an intermediate image plane before the Pockels relay, and check whether f_dc changes.
6. **Find out whether the gate depends on position or on angle.** With MB, translate the Pockels cell sideways by about
   0.5–1 mm, and separately tilt it by about 0.5–1°. Record how the per-pixel IRF changes in each case.
   - This decides whether moving the cell to an image plane would help (see below).
   - From existing data: at 40 MHz the IRF varies little across the field, which in the pupil plane means little with ray
     angle (|h| 0.37–0.48 for the 5–95 % of pixels, phase ±0.04 ns). At 20 MHz it varies strongly (|h| 0.65–1.03,
     phase −2.8 to +1.4 ns).

### Pupil plane or image plane for the Pockels cell?

- **In a pupil plane**, every ray reaching a pixel crosses the crystal at the same *angle*, but at a *position* that depends
  on where the light came from. So variations of the gate with position are not calibrated by the IRF for light whose
  pupil filling differs from MB's.
- **In an image plane** it is the other way round. Every ray reaching a pixel crosses the same crystal *position*, in focus or
  not, so the per-pixel IRF calibrates any variation with position for all light. The *angle* then depends on the pupil
  position, i.e. on the source geometry.
- **Étendue is conserved, so the crystal sees the same range of angles either way.** It just changes from field angle to
  pupil angle. For example: a 130 µm field at NA 1.49 into a 4 mm aperture gives about ±0.05 rad.
- **So an image-plane cell fixes the problem if the crystal's variation with position is much stronger than its variation
  with angle.** That is what the user's pupil map and the 40 MHz numbers above suggest. At 20 MHz the angle sensitivity is
  strong, so the image plane would be worse there. Confirm with step 6 before rebuilding.
- **Costs of an image plane:**
  - Dust and defects in the crystal are imaged sharply. Put the image plane mid-crystal, with clean faces.
  - A long crystal is only in focus at one depth, so light is blurred by about L·θ at its faces.
  - The field must fit the clear aperture.
  - The field-dependent IRF pattern becomes the crystal's spatial non-uniformity (fine, since it is calibrated per pixel).
- **The cell is two compensating crystals, not one surface,** so neither plane is exact. A ray at internal angle θ ≈ θ_air/n
  moves sideways by z·θ at distance z from the conjugate plane. In the image plane, each pixel's footprint is therefore a
  spot of radius about z_max·θ around its conjugate point, rather than the whole aperture as now. For example, with
  ±0.05 rad in air, n ≈ 2 and the image plane midway along a 40 mm pair (z_max = 20 mm), the radius is about 0.5 mm,
  against the full ~4 mm in the pupil plane. Put the image plane midway between the two crystals, not in one of them.
  The uncalibrated residual is roughly (gate change per mm from step 6) × that radius.
- **The pair's angular sensitivity is already measured.** The compensation cancels the static birefringence exactly only
  on axis. In the current pupil-plane setup pixel = angle through both crystals, so the field-dependent IRF (40 MHz:
  |h| 0.37–0.48, phase ±0.04 ns) already includes the imperfect off-axis compensation.
- **Telecentricity at the crystal matters, but not for position.** A ray reaching pixel x has angle θc(x) + Δθ, where θc
  is the chief-ray angle and Δθ is set by where it crossed the pupil. Exactly at the conjugate plane, defocus cannot move
  the position. Elsewhere in the crystal the shift is z·(θc + Δθ). The z·θc part is the same for all light at that pixel,
  so the IRF calibrates it; only z·Δθ depends on the source, telecentric or not. Telecentricity matters for angle: the
  pair's residual retardance is roughly quadratic about the axis, so with a chief-ray tilt θc a source-dependent Δθ
  changes the gate to first order (∝ θc·Δθ) instead of second order. Non-telecentric beams also walk off by z·θc, which
  uses more of the clear aperture. Make the relay telecentric at the crystal: a 4f relay with the pupil one focal length
  before the lens that forms the image in the cell.

### E4. Excitation purity (H2; run alongside E1, ~1–2 h)

1. **Look at the laser waveform.** Put a fast photodiode (≥1 GHz) and an oscilloscope on the laser at each power setting.
   Record:
   - the on/off extinction ratio;
   - the off-state level;
   - the turn-on delay;
   - the duty cycle.

   Also record the spectrum of the laser on and off (spectrometer, log scale) to look for spontaneous emission or ASE.
2. **Add a laser-line clean-up filter** to the excitation path and repeat bulk Cy5 against MB.
   - If f_dc drops, it's H2.
3. **Compare the shape of the unmodulated light with the beam profile.** Map the unmodulated light (the ring analysis in
   `rings.py`) against a defocused image of the laser beam. Coherent laser light shows rings; spontaneous emission
   doesn't.

### E5. Dye panel (H3; only if E1–E4 don't settle it)

Measure under the same conditions, with bracketing IRFs and a fixed laser current:
- MB 1 µM, on a passivated surface;
- Cy5 at 1 µM and at 50 µM;
- ATTO 655 at 1 µM;
- Alexa 647 at 1 µM;
- free bead dye, if it can be obtained.

Use two frequencies (20 MHz with full 360° coverage, and 40 MHz) to separate a long-lived component from light that is
truly unmodulated.

## Suggested schedule

| session | experiments |
|---|---|
| 1 | session rules 1–4; E1 (defocused beads); E4.1 (photodiode); E3.4 (RF-off static split); E2.1 (depth series) |
| 2 | E2.2–E2.5 (adsorption, bound dye, sandwich thickness, scatter reference) |
| 3 | E3.1–E3.3, E3.5 (pupil imaging, stops, field stop), depending on the session 1–2 results |
| 4 | E5 if still unresolved |

## What each outcome means

| result | conclusion | what to do |
|---|---|---|
| defocused beads demodulated; supercritical ring present for surface samples; the iris equalises MB and Cy5 | H1a: the gate varies across the pupil, and surface and bulk emitters fill the pupil differently | measure with a pupil iris, or calibrate with a bulk IRF (scatter or passivated dye); if E3.6 shows the variation is with position, move the Pockels cell to an image plane |
| RF-off static-split test shows light bypassing the gate in the bulk | H1b: stray light bypasses the gate | baffles, field stop, check the crystal aperture and polariser acceptance |
| the clean-up filter reduces f_dc, or the photodiode or spectrum shows an off-state or ASE floor | H2: unmodulated excitation | clean-up filter; deeper off-state; external modulator |
| bound dye is demodulated, and f_dc differs between 20 and 40 MHz | H3: photophysics or a long-lived species | change the dye or buffer (oxygen scavenging, triplet-state quenchers) |
| dark-frame offset differs from the margin estimate by several % of the sample signal | H4: offset error | dark-frame subtraction in `bulk_phasor.py` |
