# eo-flim-selvin
python code for analyzing eo-flim data (specifically for single molecules) with square wave excitation. includes pycromanager scripts too

# Pipeline
1. Use bead images to calculate registration.json (or use previous json)
2. Register the two view data with register_all.py
3. Calculate modulation depth and fit IRFs with modulation.py on the modulation data (IRF measurements) 
4. Calculate lifetime LUTs which uses G/(G+U) ratio to calculate lifetime. Makes a family of curves for different target lifetimes by varying gating function phase (g(t, phase))
5. If SMLM or bead images, run sm_lifetime.py, or superpixel_lifetime.py which does pixel-by-pixel measurement

6. For bulk samples, run bulk_phasor.py: IRF consistency (vs time, power), phasor plots, power/exposure trends and blank subtraction. Then run lifetime_estimators.py to compare estimators against shot noise.

You can also read ThunderSTORM data with ts_ratio.

See [docs/eo_flim_method_notes.md](docs/eo_flim_method_notes.md) for the method, conventions, known pitfalls and a dataset log.

Note that my DDS steps phase positively so the waveform moves BACKWARDS (phase_sign -1)