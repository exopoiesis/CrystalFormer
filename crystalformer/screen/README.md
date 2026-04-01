# Screening Pipeline

Pluggable filter pipeline for batch screening of generated crystal structures. Chain multiple criteria (validity, void size, percolation, composition, density) and get a funnel summary showing where candidates drop off.

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Python API](#python-api)
  - [ScreeningPipeline](#screeningpipeline)
  - [ScreeningResult](#screeningresult)
  - [Built-in Filters](#built-in-filters)
  - [Custom Filters](#custom-filters)
- [Command-Line Interface](#command-line-interface)
- [String Spec Format](#string-spec-format)
- [Examples](#examples)

## Installation

Same as the analysis module:

```bash
pip install crystalformer[analysis]
```

## Quick Start

### Python

```python
from crystalformer.screen import ScreeningPipeline, CompositionFilter

pipeline = ScreeningPipeline([
    "validity",
    "voronoi:r_min=1.0",
    "percolation:min_dim=2,r_probe=0.4",
    CompositionFilter(elements=["Fe", "S"]),
])

result = pipeline.run(structures)
print(result.summary())
# Screening: 5000 input → 127 passed
#   input                  5000  (100.0%)
#   validity               4812  ( 96.2%) (-188)
#   voronoi                2034  ( 40.7%) (-2778)
#   percolation             389  (  7.8%) (-1645)
#   composition             127  (  2.5%) (-262)
```

### Command line

```bash
python -m crystalformer.screen structures.csv \
    --filters validity,voronoi,percolation,composition \
    --r-probe 0.4 --percolation-dim 1 --elements Fe S \
    -o candidates.csv
```

## Python API

### ScreeningPipeline

```python
class ScreeningPipeline(filters: list)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `filters` | list | List of `Filter` instances or [string specs](#string-spec-format) |

**Methods:**

- `run(structures) -> ScreeningResult` — run all filters sequentially on a list of `pymatgen.Structure`

Filters are applied in order. Each filter sees only the survivors of the previous one. This means filter order matters for performance — put cheap filters (validity, composition) first, expensive ones (percolation) last.

### ScreeningResult

| Field | Type | Description |
|-------|------|-------------|
| `input_count` | int | Number of structures fed in |
| `survivors` | list[Structure] | Structures that passed all filters |
| `survivor_indices` | list[int] | Indices into the original input list |
| `stage_counts` | dict[str, int] | Count after each stage (ordered) |

**Methods:**

- `summary() -> str` — human-readable funnel table

### Built-in Filters

All filters inherit from `Filter` (abstract base class) and implement `__call__(structure) -> bool`.

#### ValidityFilter

Rejects structures with overlapping atoms.

```python
ValidityFilter(min_dist=0.5)  # min interatomic distance in Å
```

#### VoronoiFilter

Keeps structures whose largest void exceeds a threshold.

```python
VoronoiFilter(r_min=1.0, r_probe=0.4, grid_resolution=0.25)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r_min` | 1.0 | Minimum max-void-radius to pass (A) |
| `r_probe` | 0.4 | Probe radius for void-fraction analysis (A) |

Uses `VoronoiAnalyzer` internally.

#### PercolationFilter

Keeps structures with enough percolating channel directions.

```python
PercolationFilter(min_dim=1, r_probe=0.4, grid_resolution=0.3)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_dim` | 1 | Minimum percolation dimensionality (0–3) |
| `r_probe` | 0.4 | Probe radius in Å |

Uses `PercolationAnalyzer` internally.

#### CompositionFilter

Keeps structures that contain **all** specified elements.

```python
CompositionFilter(elements=["Fe", "S", "Ni"])
```

An empty element list passes everything.

#### DensityFilter

Keeps structures within a density range.

```python
DensityFilter(min_density=3.0, max_density=6.0)
```

Either bound can be `None` (no limit).

### Custom Filters

Subclass `Filter` and implement `__call__`:

```python
from crystalformer.screen import Filter, ScreeningPipeline

class StabilityFilter(Filter):
    """Example: user plugs in their own ML potential."""
    name = "stability"

    def __init__(self, model, threshold=0.05):
        self.model = model
        self.threshold = threshold

    def __call__(self, structure) -> bool:
        energy = self.model.predict(structure)
        return energy < self.threshold

pipeline = ScreeningPipeline([
    "validity",
    "voronoi:r_min=0.4",
    StabilityFilter(model=my_chgnet, threshold=0.05),
])
```

Custom filters integrate seamlessly — they appear in `stage_counts` under their `name` attribute and in the funnel summary.

## Command-Line Interface

```
python -m crystalformer.screen [-h] [-o OUTPUT]
                                [--filters FILTERS]
                                [--min-dist MIN_DIST]
                                [--r-min R_MIN]
                                [--r-probe R_PROBE]
                                [--percolation-dim DIM]
                                [--elements EL [EL ...]]
                                [--min-density D] [--max-density D]
                                input
```

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | — | CSV with a `cif` column |
| `-o, --output` | `<input>_screened.csv` | Output CSV (survivors only, all original columns preserved) |
| `--filters` | `validity,voronoi` | Comma-separated filter names |
| `--min-dist` | 0.5 | ValidityFilter threshold (Å) |
| `--r-min` | 1.0 | VoronoiFilter void threshold (Å) |
| `--r-probe` | 0.4 | PercolationFilter probe radius (Å) |
| `--percolation-dim` | 1 | PercolationFilter min dimensionality |
| `--elements` | — | CompositionFilter required elements |
| `--min-density` | — | DensityFilter lower bound (g/cm³) |
| `--max-density` | — | DensityFilter upper bound (g/cm³) |

## String Spec Format

For the Python API, filters can be specified as strings instead of objects:

```
"filter_name"                          → FilterClass()
"filter_name:key=val"                  → FilterClass(key=val)
"filter_name:key1=val1,key2=val2"      → FilterClass(key1=val1, key2=val2)
```

For `CompositionFilter`, elements are separated by `+`:

```
"composition:elements=Fe+S+Ni"        → CompositionFilter(elements=["Fe","S","Ni"])
```

Available names: `validity`, `voronoi`, `percolation`, `composition`, `density`.

## Examples

### Full screening workflow

```bash
# 1. Generate structures
python main.py --optimizer none --restore_path model.pkl \
    --spacegroup 194 --num_samples 5000 \
    --composition_bias "Fe:2.0,S:1.5"

# 2. Convert to CIF
python scripts/awl2struct.py --output_path . --label 194

# 3. Screen for layered ionic conductors
python -m crystalformer.screen output_194_struct.csv \
    --filters validity,voronoi,percolation,composition \
    --r-probe 0.4 --percolation-dim 2 --elements Fe S \
    -o candidates.csv

# 4. Relax survivors only (saves compute!)
python scripts/mlff_relax.py --restore_path . \
    --filename candidates.csv --model orb-v3-conservative-inf-mpa ...

# 5. Energy above hull
python scripts/e_above_hull_alex.py --restore_path . \
    --filename relaxed_structures.csv ...
```

### Comparing filter strictness

```python
from crystalformer.screen import ScreeningPipeline

for r_probe in [0.3, 0.4, 0.6, 0.8]:
    pipe = ScreeningPipeline([
        "validity",
        f"percolation:min_dim=1,r_probe={r_probe}",
    ])
    result = pipe.run(structures)
    print(f"r_probe={r_probe}: {len(result.survivors)}/{len(structures)} pass")
```

### Compositional guided sampling (upstream feature)

The `--composition_bias` flag in `main.py` adds a soft logit bias during atom-type sampling, nudging the model toward desired elements without hard-constraining it.

```bash
python main.py --optimizer none --restore_path model.pkl \
    --formula FeS --composition_bias "Fe:3.0,S:3.0,Ni:2.0" \
    --num_samples 100 --K 10
```

**Format:** `"Element:weight,Element:weight,..."`. Weights are **additive log-odds**: +2.0 roughly doubles sampling probability per step, +3.0 roughly triples it. Negative values suppress. The bias is applied *after* the atom mask, so masked-out elements stay blocked regardless of bias.

**Typical values:**

| Weight | Effect |
|--------|--------|
| 0.0 | No change (default) |
| 1.0-2.0 | Gentle nudge — increases variety with target elements |
| 3.0-5.0 | Strong bias — most atoms will be target elements |
| > 5.0 | Near-deterministic — almost only target elements |
| -1.0 to -3.0 | Suppresses an element without fully blocking it |

Combine with screening to generate-then-filter efficiently:

```bash
# 1. Biased generation
python main.py ... --composition_bias "Fe:3.0,S:3.0"

# 2. Convert
python scripts/awl2struct.py --output_path ./results/ --formula FeS

# 3. Screen for conductors
python -m crystalformer.screen results/output_FeS_struct.csv \
    --filters validity,voronoi,percolation \
    --r-min 1.0 --percolation-dim 1
```

### Reusing survivor indices

```python
result = pipeline.run(structures)

# Get original data for survivors
for idx in result.survivor_indices:
    original_row = my_dataframe.iloc[idx]
    print(original_row["formula"], original_row["spacegroup"])
```
