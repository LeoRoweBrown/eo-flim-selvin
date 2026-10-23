# pymmcore scripts

Python (pymmcore) versions of the Micro-Manager BeanShell scripts in
`C:\Users\Selvin Lab\Documents\MicroManager\scripts`, plus the Tucsen trigger tests.

Run from the `eoflim-pmm` env (pymmcore 11.9.0.73.0, matching the API_73 Micro-Manager install) with
**Micro-Manager closed**: each script loads the hardware config itself (`mmcore_common.load_core`,
~7 s) and releases it at exit. Common options: `--config`, `--mm-dir`, `--dds-port`, `--mm-log`.

| script | from | does |
|---|---|---|
| `set_phases_multichannel.py` | `set_phases_multichannel.bsh` | program the 409C phase table, per-channel settings (edit the constants at the top) |
| `rf_off.py` | – | same table with every voltage 0 (RF drive off, e.g. to test for heating); `set_phases_multichannel.py` turns it back on |
| `set_phases.py` | `set_phases.bsh` | older single-channel table |
| `acquire_phases.py` | `aquire_phases.bsh` | one frame per phase, camera trigger output steps the table; frames by software trigger as in `live_modulation_pmm.py` (checks for untriggered frames); saves a TIFF |
| `acquire_phases_notrigger.py` | `acquire_phases_notrigger.bsh` | one frame per phase, row set over serial before each snap; saves a TIFF |
| `enable_phase_stepping_live.py` / `disable_phase_stepping_live.py` | same names `.bsh` | DDS stepping on camera trigger on / off |
| `test_com12_409c.py` | `test_com12_409c.bsh` | send `Q`, print the 409C's answer |
| `test_software_trigger.py` | `test_software_trigger.bsh` | does the Tucsen give one frame per software pulse; switches to reproduce other scripts' setup |
| `tucsen_trigger_test.py`, `tucsen_dump_props.py` | – | ROI / trigger combinations; dump or diff all camera properties |
| `mmgui_tucsen.py` | – | start pymmcore-gui (`mmgui`) with the Tucsen adapter loadable (preloads its DLLs); takes mmgui's options |

`mmcore_common.py`: hardware loading (adapter folder on the DLL path, Tucsen SDK preloaded with
ctypes), the 409C serial helpers from `serialFunctions.bsh`, and `Laser` (power + enable through the
`640 power` / `640 enable` config groups, as in `live_modulation_pmm.py`). The two acquire scripts
switch the laser on for the acquisition only (`--laser-power`, `--no-laser`).

Not converted: `HamamstuSetRoi.bsh` (a single `setROI` for the Hamamatsu, only useful inside a running
session; in a pymmcore script it's one `core.setROI(180, 600, 1732, 1292)` after `load_core`) and
`serialFunctions.bsh` (now `mmcore_common.py`).
