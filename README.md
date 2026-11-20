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
| Sensitivity figure | `python3 -m ais_sim.sensitivity` | `sensitivity_*.pdf`, `sensitivity_*.csv` |
| Feasibility map and damped feasibility limit | `python3 -m ais_sim.feasibility_maps` | `feasibility_k_phi_*.pdf`, `feasibility_zeta_phi_*.csv` |
| Damped solver table | `python3 -m ais_sim.solver_report` | `solver_damped_summary_*.csv` |
| Hardware re-analysis (numbers in Sec. 5.2.1) | `python3 -m ais_sim.hardware_reanalysis` | `hardware_reanalysis_summary_*.csv` |

The FT comparison and the noise study can also be simulated from scratch by leaving out the options. The noise study then runs 1200 identifications and takes much longer. Each script prints its main numbers and lists its options with `--help`.

## Experiments (MATLAB)

MATLAB R2020a or later. The scripts find the data from their own location, so they can be run from any folder:

```
matlab -batch "run('experiments/residual_metrics.m')"
matlab -batch "run('experiments/modal_fit.m')"
```

| Script | Paper section | Output in `experiments/out/` |
| --- | --- | --- |
| `residual_metrics.m` | residual RMS and peak-to-peak angle, Sec. 5.2.3 | `residual_metrics/raw_10s_residual_angle_energy_results_gates.csv` (one row per run), the summary by length and method, and the same values divided by the median Pulse value |
| `modal_fit.m` | two-mode fits and mode-1 energy (Table 5), Sec. 5.2.3 | `modal_fit/modal_10s_results_gates_bestof10.csv` (one row per run), the summary by length and method, the log of all initializations, and `best_fit_decompositions/` with the selected fit of each run |

Both scripts use the swing angle `swing_pitch_deg` from 0.20 s to 10.20 s after the command stop, resampled at 0.01 s and demeaned. The stop is the first sample after the motion where the commanded speed falls to 5 % of its peak. `residual_metrics.m` reports the peak-to-peak value, the RMS and the integral of the squared angle over this window.

`modal_fit.m` smooths the window with a 3-sample moving mean and fits

    theta(t) = b + A1 exp(-d1 t) sin(2 pi f1 t + phi1) + A2 exp(-d2 t) sin(2 pi f2 t + phi2).

Cost. The error e is reduced by a dead band of 0.045 deg, half the 0.09 deg encoder tick, and the cost is sqrt(sum(w max(|e| - 0.045, 0)^2) / sum(w)). The weights w are 8 for the first 0.5 s, 4 up to 2 s and 1 after that, scaled to a mean of 1. If |A2| > 1.8 |A1| the cost grows by 0.03 (|A2| - 1.8 |A1|).

Bounds. |b| <= 10 deg, |A1| and |A2| <= 80 deg, 0 <= d1 <= 2 1/s, 0 <= d2 <= 40 1/s. f1 lies in [max(0.15, 0.6 fL), min(1.4, 1.5 fL)] Hz with fL = sqrt(g/L) / (2 pi) and g = 9.81 m/s^2. f2 lies between max(1.2 Hz, 1.5 times the upper f1 bound) and 20 Hz. Outside the bounds the cost is 1e6.

Fit. `fminsearch` (MaxIter 3000, MaxFunEvals 12000, TolX and TolFun 1e-9) first fits mode 1 alone and then both modes, starting from the mode 1 result.

Initializations. Each run is fitted from 10 starts. Start 1 uses f1 = fL, d1 = 0.005, phi1 = 0, A1 = half the peak-to-peak angle, and mode 2 at the strongest FFT peak of the mode 1 residual within the f2 bounds with d2 = 0.8. Starts 2 to 10 draw the frequencies, decay rates, phases, amplitudes and signs at random around these values. Start k of run i seeds the generator with `rng(1000000 + 10000*i + 100 + k)`, where i counts the runs in folder order.

Selection. The start with the lowest score is kept, where the score is the cost plus 0.10 times the weighted RMS error, 0.05 times the RMS error over the first 0.5 s and 0.02 times the RMS error over the first 2 s.

Energy. For each mode, E = L^2 thetadot^2 / 2 + g L (1 - cos theta) per unit mass, with g = 9.80665 m/s^2, is averaged over the window in J/kg. The reported mode 2 energy is zero when |A2| <= 1.25 ticks, its peak-to-peak value <= 2.5 ticks or its RMS <= 0.85 ticks. All reported energies are zero when the whole residual has a peak-to-peak value <= 2 ticks and an RMS <= 0.75 ticks.

The energy table and the experimental figures of the paper are computed from these outputs.

## Data

`data/experiments/` holds the 100 hardware runs: unshaped (Pulse), ZV, ZVD and AIS commands at rope lengths of 0.5, 0.7, 0.9, 1.1 and 1.3 m (folders `L050` to `L130`), five runs each. Each run folder has `raw.csv` with the columns the analysis reads: the motion time, the payload sample time, the commanded velocity in x and y in mm/s, the payload swing in mm and the swing angle in deg. The 50 Pulse and AIS runs also have `summary.txt` with the rope length, the command stop time and the identification values the crane logged, read by `ais_sim.hardware_reanalysis`. In the folder names, `zv` marks ZV runs, `robust` marks ZVD runs and `is2` marks AIS runs.

`data/ft_vs_ais_sweep.csv` and `data/ft_vs_ais_history.csv` are the simulated sweep and time histories behind the FT comparison figure. `data/noise_study_trials.csv` has one row per noise trial behind the noise figure and statistics.

## License

MIT, see `LICENSE`.
