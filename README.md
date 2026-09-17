# Lunar Regolith Maturity Classification

| | |
| --- | --- |
| Final rank | not ranked |
| Domain | Computer Vision |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Solutions submitted | 2 |
| Last submission | 2026-07-03 |

## Problem statement

### Overview

Build a model that classifies the space-weathering maturity stage of a lunar regolith sample from a rendered 3D close-up of a grain pile, and predicts four quantitative properties: three signed weathering-anomaly scores and a physical grain-size measurement.

Each image is a 3D render (512×512 px JPEG) built from a real, CC0-licensed Poly Haven lunar regolith 3D scan, with a procedurally generated agglutinate-glass and nanophase-iron darkening overlay painted onto the real grain mesh surfaces. The classification target is regolith maturity: immature, submature, mature, or highly mature.

The challenge has two parts:

**Classification**: Regolith maturity is conventionally quantified by the Is/FeO spectral ratio and agglutinate glass abundance, both of which increase gradually and continuously with cumulative micrometeorite-impact exposure time. There is no sharp visual transition between adjacent maturity bins — a "mature" sample differs from a "submature" sample only in degree, not in kind.

**Property regression**: Each sample requires predicting four numeric properties:

- `nanophase_iron_darkening_anomaly` — how much darker or brighter the sample is than a typical sample of its class `[-1, 1]`
- `agglutinate_glass_anomaly` — how much more or less impact glass the sample has than a typical sample of its class `[-1, 1]`
- `reflectance_anomaly_score` — signed residual between actual reflectance and the reflectance expected given the grain size `[-1, 1]`
- `mean_grain_size_microns` — mean visible grain diameter in micrometres `[40, 800]`

The three anomaly columns are within-class residuals centred at 0 — predicting them from the class label alone yields near-zero expected value. The model must estimate all four properties independently from the image.

### Evaluation

Submissions are scored with a weighted composite metric remapped to `[-1, 1]`. Higher is better.

```
raw  = 0.60 * macro_f1 + 0.40 * mean(1 - normalized_RMSE per column)
score = 2 * raw - 1
```

A perfect submission scores `1.0`. A completely wrong submission (zero F1, maximum RMSE on every column) scores `-1.0`. Random or uninformed predictions typically score near `-0.5` to `0.0`.

Each column's RMSE is normalized by its full value range before averaging:

- Signed anomaly columns range `[-1, 1]` → range = 2.0
- `mean_grain_size_microns` ranges `[40, 800]` → range = 760.0

The observation component carries 40% of the raw score. A model that predicts 0.0 for all anomaly columns will incur significant RMSE on samples with large-magnitude anomalies, pulling the final score down.

### Dataset

The prepared public data has 2,400 training images and 800 test images. Images are 512×512 px JPEGs rendered under randomised close-up camera angles with low-angle grazing illumination.

```
public/
├── train.csv
├── test.csv
├── sample_submission.csv
└── images/
    ├── regolith_train_00000.jpg
    ├── ...
    └── regolith_test_00799.jpg
```

`train.csv` columns:

| Column | Type | Range | Description |
| --- | --- | --- | --- |
| image | string | — | Relative path to the image file, e.g. `images/regolith_train_00000.jpg` |
| label | int | 0 – 3 | Maturity class (0 = immature, 1 = submature, 2 = mature, 3 = highly mature) |
| nanophase_iron_darkening_anomaly | float | [-1, 1] | Signed within-class darkening deviation |
| agglutinate_glass_anomaly | float | [-1, 1] | Signed within-class glass-fraction deviation |
| reflectance_anomaly_score | float | [-1, 1] | Signed reflectance residual vs. grain-size expectation |
| mean_grain_size_microns | float | [40, 800] | Mean visible grain size in micrometres |

`test.csv` columns:

| Column | Type | Description |
| --- | --- | --- |
| image | string | Relative path to the image file, e.g. `images/regolith_test_00000.jpg` |

### Submission

Submit a CSV file named `submission.csv` with one row per row in `test.csv`. Example:

```
image,prediction,nanophase_iron_darkening_anomaly,agglutinate_glass_anomaly,reflectance_anomaly_score,mean_grain_size_microns
images/regolith_test_00000.jpg,2,0.142,-0.317,0.085,312.5
images/regolith_test_00001.jpg,0,-0.503,0.211,-0.649,587.3
images/regolith_test_00002.jpg,3,0.761,0.488,-0.104,98.7
```

Full column specification:

| Column | Type | Range | Description |
| --- | --- | --- | --- |
| image | string | — | Image path copied exactly from `test.csv` |
| prediction | int | 0 – 3 | Predicted maturity class |
| nanophase_iron_darkening_anomaly | float | [-1, 1] | Predicted darkening anomaly |
| agglutinate_glass_anomaly | float | [-1, 1] | Predicted glass-fraction anomaly |
| reflectance_anomaly_score | float | [-1, 1] | Predicted reflectance anomaly |
| mean_grain_size_microns | float | [40, 800] | Predicted grain size in micrometres |

Requirements:

- Exactly one row per image in `test.csv`, in any order
- Header row must be present
- No duplicate image paths
- `prediction` must be an integer (0, 1, 2, or 3)
- The three anomaly columns must be in `[-1, 1]` — values outside this range are rejected
- `mean_grain_size_microns` must be in `[40, 800]`
