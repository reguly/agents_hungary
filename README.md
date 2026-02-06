# Hungarian Census Synthetic Population Generator

This repository builds a synthetic micro-population from Hungarian census aggregates.
The implementation is in `synth_pop.py` and is designed for large runs (national scale).

## Goals

The pipeline generates person-level, household-level, and dwelling-level records that are consistent with:

- settlement-level flat marginals (population, household, housing)
- county-by-settlement-type hierarchical conditionals (activity, schooling, occupation, sector, commuting, health, household structure)

The output is intended for simulation and policy analysis where full microdata is not available.

## Data Inputs

The script expects these files in the same directory as `synth_pop.py`:

- `flat_A_nepesseg_adatok_telepulesenkent.xlsx`
- `flat_A_haztartasok_adatai_telepulesenkent.xlsx`
- `flat_A_lakasok_legfontosabb_jellemzok_telepulesenkent.xlsx`
- `hier_gazdasagi_aktivitas_varmegyenkent_telepulestipusonkent.xlsx`
- `hier_foglalkoztatott_nepesseg_foglalkozasi_focsoport.xlsx`
- `hier_foglalkoztatott_nemzetgazdasagi_agazat.xlsx`
- `hier_foglalkoztatott_ingazas.xlsx`
- `hier_iskolaba_jaro_nepesseg.xlsx`
- `hier_Egeszseg_allapot_varmegyenkent_telepulestipusonkent.xlsx`
- `hier_nem_kor_csaladiallapot.xlsx`
- `hier_nem_kor_gazdasagiaktivitas_iskolaivegzettseg.xlsx`
- `hier_homesize_agegroup_employment.xlsx`
- `telepules_hierarchia.xlsx`

## Execution

### Requirements

- Python 3.10+
- `numpy`
- `pandas`
- `pyarrow`
- Excel read support via `openpyxl`

Install example:

```bash
pip install numpy pandas pyarrow openpyxl
```

### Command line

```bash
python synth_pop.py --output-dir output_country --diagnostics
```

Options:

- `--output-dir`: output folder (default `output`)
- `--max-settlements`: process only first N settlements
- `--county`: restrict to one county (`varmegye`)
- `--seed`: RNG seed
- `--skip-households`: skip household and dwelling generation
- `--diagnostics`: generate diagnostic workbooks from synthetic output

## Outputs

The pipeline writes these files to `--output-dir`:

- `synthetic_population.parquet`
- `synthetic_households.parquet` (unless `--skip-households`)
- `synthetic_dwellings.parquet` (unless `--skip-households`)
- `validation_report.json`
- `diagnostics/*.xlsx` (if `--diagnostics`)

### Person schema

`synthetic_population.parquet` columns:

- `person_id`
- `household_id`
- `settlement`
- `ksh_code`
- `varmegye`
- `telepules_tipus`
- `sex`
- `age`
- `age_group_5y`
- `age_group_10y`
- `education`
- `employment_status`
- `occupation_group`
- `sector`
- `commute_region`
- `commute_type`
- `school_level`
- `marital_status`
- `children_count`
- `disability_status`
- `chronic_status`
- `limitation_status`

### Household schema

`synthetic_households.parquet` columns:

- `household_id`
- `settlement`
- `size`
- `family_structure`
- `age_composition_flags`
- `employment_composition`
- `dwelling_id`

### Dwelling schema

`synthetic_dwellings.parquet` columns:

- `dwelling_id`
- `rooms`
- `settlement`

## Process Overview

The workflow is deterministic for a fixed seed and follows these stages.

### 1. Parse and normalize source tables

- Flat tables are loaded as settlement-wide category counts.
- Hierarchical tables are parsed from multi-row headers and converted to tidy long format.
- Budapest "unassigned district" counts are proportionally redistributed to district columns.

### 2. Build settlement crosswalk

`telepules_hierarchia.xlsx` is used to map each settlement to:

- `ksh_code`
- `varmegye`
- `telepules_tipus`

This enables joining settlement-level marginals to county-by-type conditionals.

### 3. Build seed tensors from hierarchical activity tables

Core tensor dimensions:

- sex
- age band
- education
- employment status

Two sources are used:

- full activity hierarchy (`hier_gazdasagi_aktivitas...`)
- nem-kor activity+education hierarchy (`hier_nem_kor_gazdasagiaktivitas_iskolaivegzettseg...`)

The script builds county+type seed arrays and global fallback arrays.

### 4. Settlement-level core fit with IPF

For each settlement:

- derive population total and flat marginals
- construct seed joint distribution from county+type hierarchy
- apply iterative proportional fitting (IPF) to match local marginals
- integerize cell counts with largest-remainder allocation

This yields person counts by `(sex, age-band, education, employment)`.

### 5. Single-year age allocation

The pipeline converts grouped age counts to single-year ages (`0..100`), including:

- decade splits driven by county+type profiles
- explicit handling for upper age bins (`70-79`, `80-89`, `90+`)
- under-7 / under-15 consistency adjustments

### 6. Secondary person attributes

Attributes are assigned from conditional hierarchies, with fallback to broader distributions when sparse:

- occupation group (employed only)
- sector (employed only)
- commuting region and type (employed only)
- school level (with inferred non-attendance)
- marital status (15+ constraints applied)
- fertility category (women 15+)
- health states (disability/chronic/limitation)

### 7. Household generation

Household synthesis combines model-based totals and constrained assignment:

- build 3D household target table over:
  - `size`
  - `age_comp`
  - `employment_comp`
- use 3D IPF with `hier_homesize_agegroup_employment.xlsx` seed
- instantiate household specs from integerized 3D counts
- assign persons to households under age and employment compatibility rules

### 8. Household post-calibration by swaps

A greedy swap pass refines difficult marginals for:

- unemployed count bins per household
- inactive count bins per household
- dependent count bins per household

Swaps preserve age bucket composition to avoid degrading age-composition fit while improving employment-composition fit.

### 9. Dwelling synthesis and matching

From flat housing marginals:

- create occupied dwelling stock by room category
- assign one dwelling per household
- if dwelling stock is short, add fallback dwellings (`rooms=2`)

### 10. Validation and diagnostics

The run emits:

- per-settlement validation deltas (`validation_report.json`)
- reconstructed flat diagnostics (`orig/gen/diff%`)
- reconstructed hier diagnostics (`orig/gen/diff%`) with interleaved columns

## Algorithmic Choices

### IPF for marginal consistency

IPF is used because it is stable, scalable, and preserves seed structure while forcing marginals to match.

### Integerization

Fractional expected counts are integerized with largest-remainder logic while preserving grand totals.

### Hierarchical fallback strategy

When a county+type conditional is sparse or unavailable:

- fall back to broader county+type marginals
- fall back to global distribution if still empty

This prevents dead cells while keeping the run complete.

### Structural and invalid combinations

Hierarchical tables include structurally invalid combinations (blank cells). The implementation treats these as unavailable probability mass and relies on seed support plus domain constraints (age gates, 15+ restrictions, employed-only attributes) to avoid implausible assignments.

### Computational strategy

- settlement-wise processing to limit memory growth
- parquet streaming writers (append by settlement)
- vectorized numpy/pandas operations for band mapping and grouped assignment
- optional county/subset execution for iterative tuning

## Diagnostics Interpretation

Each diagnostic workbook compares source and reconstructed values:

- `orig`: source statistic
- `gen`: reconstructed from synthetic records
- `diff%`: `(gen-orig)/orig * 100`

For hierarchical diagnostics, column groups are interleaved by region/type:

- `varmegye | telepules_tipus | orig`
- `varmegye | telepules_tipus | gen`
- `varmegye | telepules_tipus | diff%`

This layout makes side-by-side inspection faster and avoids metric grouping drift.

## Known Limitations

- Family-structure household categories are only approximately represented.
- Sparse hier cells can create very large percentage errors when source `orig` is small.
- Some rare high-order household combinations remain hard to fit exactly.
- If `orig == 0`, percentage diff is undefined and represented as `NaN`.

## Practical Tuning Workflow

For iterative improvement:

1. Run one county with diagnostics.
2. Inspect worst categories in flat and hier outputs.
3. Adjust seed support, constraints, or post-swap limits.
4. Re-run full country after county-level convergence.

## File-Level Documentation in Code

`synth_pop.py` now includes function-level docstrings for all top-level functions, covering:

- parsing behavior
- fitting and integerization
- assignment logic
- diagnostics reconstruction
- end-to-end orchestration

These docstrings are the authoritative in-code reference for implementation details.
