# Adaptive Input Shaping

Code and data for the paper "Online Adaptive Input Shaping for Gantry Cranes with Unknown Payload Swing Dynamics". It identifies the payload swing during a short window and shapes the rest of the move to cancel the residual vibration.

This work builds on earlier code by Nyi Nyi Aung, https://github.com/NyiNyi-14/A-TDF.

## Install

Python 3.10.

```
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

Run each command from the repository root. Outputs are written to `ais_sim/out/` with a time stamp in the file name.

| Paper figure or table | Command | Output |
| --- | --- | --- |
| Representative cases: table and figure of time histories | `python3 -m ais_sim.table1` | `table1_*.csv`, `table1_fig2_*.pdf` |
| Damping sweep figure | `python3 -m ais_sim.sweeps --only zeta` | `sweep_zeta_*.pdf`, `sweeps_*.csv` |
| Sweep results over damping, frequency and window length | `python3 -m ais_sim.sweeps` | `sweeps_*.csv` |
| Near-critical damping table | `python3 -m ais_sim.near_critical` | `near_critical_*.csv` |
| Step-wise reference figure | `python3 -m ais_sim.step_wise` | `step_wise_*.pdf` |
| Comparison with the FT shaper | `python3 -m ais_sim.ft_vs_ais_paper --replot data/ft_vs_ais_sweep.csv --history data/ft_vs_ais_history.csv` | `ft_vs_ais_paper_*.pdf` |
| Measurement noise figure and statistics | `python3 -m ais_sim.noise_study --replot data/noise_study_trials.csv` | `noise_study_*.pdf`, `noise_study_*_summary.csv` |
| Sensitivity table and figures | `python3 -m ais_sim.sensitivity` | `sensitivity_*.pdf`, `sensitivity_surface_*.pdf`, `sensitivity_*.csv` |
| Feasibility map and damped feasibility limit | `python3 -m ais_sim.feasibility_maps` | `feasibility_k_phi_*.pdf`, `feasibility_zeta_phi_*.csv` |
| Damped solver table | `python3 -m ais_sim.solver_report` | `solver_damped_summary_*.csv` |
| Hardware re-analysis table | `python3 -m ais_sim.hardware_reanalysis` | `hardware_reanalysis_summary_*.csv` |

The FT comparison and the noise study can also be simulated from scratch by leaving out the options. The noise study then runs 1200 identifications and takes much longer. Each script prints its main numbers and lists its options with `--help`.

## Data

`data/hardware/` holds the 50 hardware runs analysed by `ais_sim.hardware_reanalysis`, 25 with the adaptive shaper and 25 pulse runs, at rope lengths of 0.5 to 1.3 m (folders `L050` to `L130`). Each run folder has `raw.csv` with the time stamps and the payload swing in mm, and `summary.txt` with the rope length, the command stop time and the identification values the crane logged.

`data/ft_vs_ais_sweep.csv` and `data/ft_vs_ais_history.csv` are the simulated sweep and time histories behind the FT comparison figure. `data/noise_study_trials.csv` has one row per noise trial behind the noise figure and statistics.

## License

MIT, see `LICENSE`.
