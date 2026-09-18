# `t-channel_plotting_scripts`

Plotting utilities for the CMS **EXO-24-010** search for semi-visible jets. This repository contains scripts used to produce publication-quality **two-dimensional limit plots** and **prefit/postfit distributions** for the supervised (ParticleNet, `PNET`) and unsupervised (WNAE) analyses.

## Repository layout

```text
.
├── limits/
│   ├── limitPlot2D-griddata_interpolation.py  # 2D limit-plot production
│   ├── make_2D_limit_plots.sh                 # Example limit-plot commands
│   ├── roberto_plotting_script.py             # Auxiliary plotting utilities
│   ├── signal_cross_sections.csv              # Signal cross-section inputs
│   └── unblinded_*.csv                        # Limit-scan inputs
└── postfits/
    ├── postfit_plotter.py                     # Prefit/postfit plot production
    ├── postfit_pnet_wnae.sh                   # Example PNET/WNAE commands
    ├── PNET/                                  # ParticleNet inputs/outputs
    ├── WNAE/                                  # WNAE inputs/outputs
    ├── plots_pnet/                            # ParticleNet plots
    └── plots_wnae/                            # WNAE plots
```

## Requirements

The scripts are intended for a CMS software analysis environment with Python 3. The exact environment used to produce the EXO-24-010 plots should be preferred. The main dependencies are:

- Python 3
- NumPy
- pandas
- SciPy
- Matplotlib
- `mplhep`
- CERN ROOT with PyROOT

A minimal Python installation can be prepared with:

```bash
python3 -m pip install numpy pandas scipy matplotlib mplhep
```

> `postfits/postfit_plotter.py` requires ROOT/PyROOT and ROOT fit-diagnostic files. Install or load ROOT through the appropriate CMS/analysis environment rather than relying only on pip.

## 2D limit plots

The `limits/` scripts read limit-scan CSV files containing the expected and observed limit quantiles. They interpolate the scan and produce plots with:

- The expected central limit and its ±1σ band
- The observed limit contour
- A logarithmic color map of the expected cross-section-normalized limit
- Optional scan-grid points and observed-point diagnostics
- Optional theoretical cross-section weighting from `signal_cross_sections.csv`

Change into the limits directory before running the scripts:

```bash
cd limits
```

The provided driver script contains example commands for the PNET and WNAE scans:

```bash
bash make_2D_limit_plots.sh
```

A direct invocation has the following general form:

```bash
python3 limitPlot2D-griddata_interpolation.py \
  --output-csv unblinded_gapVeto_perCateYearNonclosure_PNET-2.csv \
  --poi-x mMed \
  --poi-y rinv \
  --fixed mDark=20 yukawa=1 \
  --xsec-csv signal_cross_sections.csv \
  --legend-position "upper right" \
  --output-suffix PNET_mDark20_yukawa1
```

Other supported scan axes include `mDark` and `yukawa`. For example:

```bash
python3 limitPlot2D-griddata_interpolation.py \
  --output-csv unblinded_gapVeto_perCateYearNonclosure_WNAE-2.csv \
  --poi-x mMed \
  --poi-y mDark \
  --fixed rinv=0.3 yukawa=1 \
  --tagger wnae \
  --output-suffix WNAE_rinv0p3_yukawa1
```

Useful options include:

```text
--observed true|false|Dummy
--flag-outliers
--annotate-outliers
--overlay-scan-grid
--triangulation delaunay|rectangular|scan
--rectangular-diagonal bl-tr|tl-br
--smooth
--colorbar-range MIN MAX
--output-pdf OUTPUT.pdf
```

The script writes a PDF unless `--output-pdf` is specified explicitly. Parameter values such as `rinv=0.3` may also be written using the repository's filename convention, `rinv=0p3`.

## Prefit and postfit plots

`postfits/postfit_plotter.py` creates stitched ABCD-region plots from Combine `FitDiagnostics` ROOT files and a corresponding datacard/workspace. It can produce:

- Prefit distributions
- Background-only postfit distributions (`fit_b`)
- Signal-plus-background postfit distributions (`fit_s`)
- Data-to-background ratio panels
- PNET and WNAE category plots

The command-line interface requires one `FitDiagnostics` ROOT file, a datacard, a workspace, an output directory, a tagger, and a year:

```bash
cd postfits

python3 postfit_plotter.py \
  --input-files PNET/fitDiagnosticscombinedFitDiagnostics_mMed2000_mDark20_rinv0p3_yukawa1_Run2.root \
  --datacard PNET/combinedFitDiagnostics_mMed2000_mDark20_rinv0p3_yukawa1_Run2.txt \
  --workspace-file PNET/combinedFitDiagnostics_mMed2000_mDark20_rinv0p3_yukawa1_Run2.root \
  --output-folder plots_pnet \
  --tagger Supervised \
  --year Run2 \
  --signal-point mMed2000_mDark20_rinv0p3_yukawa1
```

For WNAE, use the corresponding WNAE inputs and select the unsupervised tagger:

```bash
python3 postfit_plotter.py \
  --input-files WNAE/fitDiagnosticscombinedFitDiagnostics_mMed2000_mDark20_rinv0p3_yukawa1_Run2.root \
  --datacard WNAE/combinedFitDiagnostics_mMed2000_mDark20_rinv0p3_yukawa1_Run2.txt \
  --workspace-file WNAE/combinedFitDiagnostics_mMed2000_mDark20_rinv0p3_yukawa1_Run2.root \
  --output-folder plots_wnae \
  --tagger Unsupervised \
  --year Run2 \
  --signal-point mMed2000_mDark20_rinv0p3_yukawa1
```

The output directory is populated with subdirectories such as:

```text
plots_pnet/
├── prefit_stitched/
├── postfit_bonly_stitched/
└── postfit_sb_stitched/
```

Only the postfit products available in the input ROOT file are produced. For example, `postfit_bonly_stitched` requires `fit_b`, while `postfit_sb_stitched` requires `fit_s`.

## Input conventions

The plotting scripts assume the following conventions:

- Limit CSV files contain the scan parameters and the columns `expected_m2sigma`, `expected_m1sigma`, `expected`, `expected_p1sigma`, `expected_p2sigma`, and `obs_lim`.
- Signal parameter tokens use `p` for decimal points, for example `rinv0p3`.
- Cross-section tables contain a `cross_section` column and matching signal-parameter columns.
- FitDiagnostics ROOT files contain the standard Combine objects used by the plotting script, including `prefit`, `fit_b`, and/or `fit_s` shapes where available.
- Datacards contain the ABCD channel and shape mappings used to identify regions A, B, C, and D.

## Reproducibility notes

- Run the scripts from the directory containing their relative input paths, or update the paths in the shell scripts.
- Keep the limit CSV, cross-section CSV, datacard, workspace, and ROOT fit outputs together with the analysis version that produced them.
- Record the exact CMSSW/ROOT and Python environments used for final paper plots.
- Inspect the generated PDFs before inclusion in a paper, especially for interpolation boundaries, physical `mMed`–`mDark` constraints, and flagged observed scan points.

## Analysis

These scripts support the CMS EXO-24-010 search for semi-visible jets. They are analysis-specific plotting tools and are not intended as a general-purpose plotting package.
