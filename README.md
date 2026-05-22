# Solar Disturbance SciML

Code accompanying the paper **"Hybrid Modeling and Estimation of Unknown Solar
Disturbances in Smart Building Thermal Systems"** (Pandya, Singh, Guo, Kong).

The repository implements a Scientific Machine Learning (SciML) framework that
combines a first-principles room heat-balance ODE with a lightweight
radial-basis-function (RBF) representation of the cloud-free solar profile and
a sigmoid-bounded latent cloud-cover state. An Unscented Kalman Filter (UKF)
jointly estimates indoor temperature and the latent disturbance from noisy
indoor-temperature measurements.

## Models compared

| ID | Model | Solar treatment | Cloud treatment |
|----|---------------------------------------|-----------------|-----------------|
| M1 | Physics-only ODE                      | `I_s = 0`       | —               |
| M2 | Geometric clear-sky, no cloud         | analytical      | `c = 0`         |
| M3 | Learned RBF clear-sky, no cloud       | learned (RBF)   | `c = 0`         |
| M4 | **Proposed UKF with latent cloud**    | learned (RBF)   | latent, UKF-estimated |

## Repository layout

```
.
├── shared_v3.jl                   # core utilities: RC simulator, RBF, UKF, metrics
├── 00_fetch_chicago_weather.jl    # downloads Chicago Open-Meteo data → data/chicago_weather.csv
├── 01_track_a_simulation.jl       # Track A: synthetic ground truth, 20 seeds
├── 02_track_b_realweather.jl      # Track B: real Chicago weather, 10 measurement-noise seeds
├── 03_make_figures.jl             # publication figures (PNG + PDF)
├── 04_paper_tables.jl             # IEEE-style LaTeX tables
├── Project.toml / Manifest.toml   # Julia environment lock
└── outputs/                       # CSVs, figures, LaTeX tables produced by the scripts
```

## Requirements

- Julia 1.10 or newer
- Internet access for `00_fetch_chicago_weather.jl` (Open-Meteo public archive)

## Reproducing the paper results

```bash
git clone <repo-url>
cd solar-disturbance-sciml

# Install Julia dependencies (one-time, ~1 minute)
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 1. Fetch Chicago Open-Meteo weather (only needed once; idempotent)
julia --project=. 00_fetch_chicago_weather.jl

# 2. Run experiments
julia --project=. 01_track_a_simulation.jl      # ≈ 1 minute, 20 seeds
julia --project=. 02_track_b_realweather.jl     # ≈ 30 seconds, 10 seeds

# 3. Render figures and LaTeX tables
julia --project=. 03_make_figures.jl
julia --project=. 04_paper_tables.jl
```

All output CSVs, PNGs, and `.tex` table fragments land in `outputs/`. The
`outputs/table_*.tex` and `outputs/fig_*.png` files are designed to be
`\input{}` and `\includegraphics{}` from the paper sources.

## Key results

On the synthetic Track A (mean ± 95% CI across 20 seeds, 3-day test):

| Model | T RMSE [°C]        | I_s RMSE [W/m²]      | Cloud RMSE         |
|-------|--------------------|----------------------|--------------------|
| M1    | 3.38 [3.01, 3.75]  | —                    | —                  |
| M2    | 3.18 [2.81, 3.55]  | 175.5 [156, 195]     | —                  |
| M3    | 3.01 [2.64, 3.37]  | 159.3 [140, 179]     | —                  |
| **M4**| **0.14 [0.135, 0.144]** | **39.8 [38.0, 41.7]** | **0.142 [0.132, 0.151]** |

On Track B (real Chicago weather, 10 noise seeds):

| Model | T RMSE [°C]        | I_s RMSE [W/m²]      |
|-------|--------------------|----------------------|
| M1    | 3.86               | —                    |
| M2    | 2.53               | 152.3                |
| M3    | 2.50 [2.40, 2.60]  | 144.4 [138, 151]     |
| **M4**| **0.20 [0.19, 0.21]** | **105.0 [101, 109]** |

All paired Diebold-Mariano comparisons against M4 give p < 1e-7.

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{pandya2026hybrid,
  author    = {Pandya, Mit and Singh, Prince Chandra and Guo, Yanhui and Kong, Liang},
  title     = {Hybrid Modeling and Estimation of Unknown Solar Disturbances in Smart Building Thermal Systems},
  booktitle = {Proc. IEEE Conf. (forthcoming)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Data attribution

Real Chicago weather data is fetched live from
[Open-Meteo](https://open-meteo.com/), licensed CC-BY 4.0
(Zippenfenig, *Open-Meteo.com Weather API*, Zenodo 2023,
DOI [10.5281/zenodo.7970649](https://doi.org/10.5281/zenodo.7970649)).
