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
  - Exposure-time effects: beads at 100 ms and 1 µM MB at 20–50 ms are fine.
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

H1 is the leading hypothesis. The plan tests it first and most directly, while running the cheap checks for H2 and H4 in
parallel.

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
