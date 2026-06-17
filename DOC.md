# DOC.md

## Project Overview

AZT1D Dashboard is a Type 1 Diabetes data analysis platform for CGM (Continuous Glucose Monitoring) data from 25 subjects using Automated Insulin Delivery (AID) systems. The project includes data loading, visualization, and future support for counterfactual analysis ("what-if" scenarios).

**Technology Stack**: Python, Streamlit, Pandas, Plotly, DVC (Data Version Control), pytest

## Essential Commands

For full environment setup, running the dashboard, and the main test commands,
see the **Installation**, **Usage**, and **Testing** sections in `README.md`.

Below are a few additional developer-oriented convenience commands:

```bash
# Run a specific test by name (useful during development)
python -m pytest tests/test_data_loader.py::TestSingleSubjectLoading::test_load_subject_returns_dataframe -v

# Inspect DVC status and raw data files
dvc status
ls -R data/raw/CGM\ Records/
```

## Code Architecture

### Data Layer (`src/data_loader.py`)
The core data loading module with robust validation and preprocessing:

- **`SubjectDataLoader`**: Main class for loading and processing subject data
  - `load_subject(subject_id)`: Load single subject with validation and derived features
  - `load_multiple_subjects(subject_ids)`: Batch load multiple subjects
  - `get_available_subjects()`: Discover available subject IDs from filesystem
  - `get_subject_summary(subject_id)`: Generate clinical statistics without full data load

- **Convenience Functions**: Top-level functions for quick access
  - `load_subject_data(subject_id)`: Quick single subject load
  - `load_all_subjects()`: Load all 25 subjects in one call
  - `get_available_subjects(data_dir)`: Get subject list

- **Validation**: Schema validation happens automatically unless `validate=False`
  - Checks for required columns (EventDateTime, CGM, Basal, etc.)
  - Validates CGM data presence
  - Handles Subject 14's known data issues gracefully

- **Derived Features**: Automatically added unless `add_derived_features=False`
  - `glucose_zone`: Categorical (severe_hypo, hypo, in_range, hyper, severe_hyper)
  - `is_hypoglycemic`, `is_hyperglycemic`, `in_target_range`: Binary indicators
  - `has_bolus`, `has_carbs`: Event indicators

### Dashboard Layer (`app.py`)
Streamlit web application with three tabs:

- **Tab 1 - Glucose Visualization**: Interactive time-series plots with Plotly
  - Multi-panel layout: CGM on top, insulin/carbs on bottom
  - Clinical thresholds overlaid (70 mg/dL hypo, 180 mg/dL hyper)
  - Date range filtering (default: 3 days for performance)

- **Tab 2 - Clinical Metrics**: Summary statistics and performance indicators
  - Time in Range (TIR), mean glucose, CV (coefficient of variation)
  - Insulin delivery totals, meal events

- **Tab 3 - Parameter Adjustment**: Placeholder for future counterfactual analysis

### Testing Layer (`tests/test_data_loader.py`)
Comprehensive test suite with 49 passing tests:

- **Test Categories**:
  - Initialization and configuration
  - Subject discovery and availability
  - Single/multiple subject loading
  - Schema validation and error handling
  - Data preprocessing and derived features
  - Integration tests with real data
  - Performance benchmarks

- **Fixtures**: Mock data generation via `temp_data_dir` fixture for isolated testing

### Data Structure
```
data/
└── raw/                    # DVC-tracked (raw.dvc)
    ├── CGM Records/
    │   ├── Subject 1/
    │   │   └── Subject 1.csv
    │   ├── Subject 2/
    │   │   └── Subject 2.csv
    │   └── ... (up to Subject 25)
    └── Visual Statistics/  # Pre-generated charts (not loaded by code)
```

**Important**: Data is NOT in git. Use `dvc pull` to fetch. Total size: ~875MB.

## Clinical Domain Knowledge

### Clinical Thresholds (ADA Guidelines)
Constants defined in `SubjectDataLoader`:
- **Severe Hypoglycemia**: < 54 mg/dL (`SEVERE_HYPOGLYCEMIA_THRESHOLD`)
- **Hypoglycemia**: < 70 mg/dL (`HYPOGLYCEMIA_THRESHOLD`)
- **Target Range**: 70-180 mg/dL (`TARGET_RANGE_LOWER`, `TARGET_RANGE_UPPER`)
- **Hyperglycemia**: > 180 mg/dL (`HYPERGLYCEMIA_THRESHOLD`)
- **Severe Hyperglycemia**: > 250 mg/dL (`SEVERE_HYPERGLYCEMIA_THRESHOLD`)

### Key Metrics
- **Time in Range (TIR)**: Percentage of readings in 70-180 mg/dL (goal: >70%)
- **CV (Coefficient of Variation)**: Glucose variability (goal: <36%)
- **CGM Frequency**: Readings every 5 minutes (~11,000 per subject per 38 days)

### CSV Schema
Expected columns (validated by `EXPECTED_COLUMNS`):
```
EventDateTime, DeviceMode, BolusType, Basal, CorrectionDelivered,
TotalBolusInsulinDelivered, FoodDelivered, CarbSize, CGM
```

## Known Issues and Quirks

### Subject 14 Data Issue
Subject 14 originally had a column named `'Readings (CGM / BGM)'` instead of `'CGM'`. A correction script exists at `scripts/fix_subject14.py` to fix this schema inconsistency. The data loader will fail validation for Subject 14 if this hasn't been corrected.

To fix Subject 14:
```bash
python scripts/fix_subject14.py
```

The script creates a backup before modification and runs quality checks.

### DVC Dependency
**Critical**: All code that loads data requires `dvc pull` to have been run first. If data is missing, you'll get a `DataLoadError` with message:
```
Data directory not found: data/raw/CGM Records
Please ensure DVC data is pulled: dvc pull command
```

### Performance Considerations
- Loading all 25 subjects (~275,000 records) takes a few seconds
- Streamlit uses `@st.cache_data` and `@st.cache_resource` decorators for performance
- Default dashboard view is 3 days to keep plots responsive

## Development Patterns

### Adding New Features to Data Loader
When extending `SubjectDataLoader`:
1. Add new derived features in `_add_derived_features()`
2. Update `EXPECTED_COLUMNS` if schema changes
3. Write unit tests in `tests/test_data_loader.py`
4. Run full test suite to ensure no regressions

### Adding New Dashboard Visualizations
Streamlit caching is critical for performance:
```python
@st.cache_data
def load_subject_data(subject_id):
    loader = load_data_loader()
    return loader.load_subject(subject_id)
```
Always cache data loading functions to avoid redundant file I/O.

### Error Handling Philosophy
The codebase uses a custom `DataLoadError` exception for all data loading failures. Always catch and log specific errors rather than using bare `except:` blocks. See `load_multiple_subjects()` for the pattern of graceful degradation (logs failed subjects but continues).

### Testing Philosophy
- **Unit tests**: Use `temp_data_dir` fixture with mock data
- **Integration tests**: Mark with `integration` and use real data from DVC
- **Coverage goal**: Current coverage is high (~90%+), maintain this standard

## Future Work

### Phase 2 - Dashboard Enhancements (In Progress)
Currently tab 1 and tab 2 are functional. Tab 3 (Parameter Adjustment) now
implements *metric-level* counterfactuals (see below) and can be extended to
therapy-level simulations once an explicit predictive model is in place.

### Phase 3 - Counterfactual Analysis (Planned)
The ultimate goal is a "what-if" simulator:
- Adjust meal timing/size
- Change insulin delivery parameters
- Predict glucose impact using ML models
- This will require scikit-learn or similar for predictive modeling

When implementing counterfactual features:
- Keep simulation logic in a new `src/simulator.py` module
- Maintain separation between data loading and simulation
- Add new test file `tests/test_simulator.py`

## Counterfactual Metrics Module (`src/counterfactual.py`)

### Scope and Philosophy

The current counterfactual implementation is intentionally conservative. We
focus on *metric-level* counterfactuals that are fully identified from the
observed CGM time series without introducing opaque physiological models.

Concretely, given a glucose trajectory :math:`g_1, \dots, g_N` (mg/dL) and a
target range :math:`[L, U]`, we define

- :math:`\mathrm{TIR}(L, U) = 100 \cdot \frac{1}{N} \sum_{t=1}^N \mathbf{1}(L \le g_t \le U)`
- :math:`\mathrm{TBR}(L)    = 100 \cdot \frac{1}{N} \sum_{t=1}^N \mathbf{1}(g_t < L)`
- :math:`\mathrm{TAR}(U)    = 100 \cdot \frac{1}{N} \sum_{t=1}^N \mathbf{1}(g_t > U)`

which follow the International Consensus on Time in Range for CGM data.

### Implemented Counterfactual Families

1. **Target-range redefinition**
   - Parameters: lower and upper bounds :math:`(L, U)` in mg/dL.
   - Transformation: keep :math:`g_t` fixed, change :math:`(L, U)` and
     recompute TIR/TBR/TAR.
   - Interpretation: "If my clinical target range were [L, U] instead of
     [70, 180], how would my summary metrics change for this subject and
     date range?"

2. **Uniform CGM bias (sensor calibration)**
   - Parameter: bias :math:`b` in mg/dL.
   - Transformation: :math:`g'_t = g_t + b` for all :math:`t`, followed by
     recomputation of TIR/TBR/TAR under a chosen :math:`[L, U]`.
   - Interpretation: "If the CGM were biased by +/−b mg/dL, what TIR/TBR/TAR
     would have been *reported* under this target range?"

These families are intentionally modest: they explore how clinical metrics
respond to changes in reporting conventions and simple measurement error,
without asserting how insulin dosing or behavior would have changed.

### API Surface

- `RangeMetrics`: small dataclass bundling `mean_glucose`, `tir`, `tbr`, `tar`.
- `compute_range_metrics(glucose, lower, upper)`: core functional that
  implements the formulas above (NaNs dropped; invalid bounds rejected).
- `apply_uniform_offset(glucose, offset_mgdl)`: returns `g + b` element-wise.
- `compute_offset_counterfactual_metrics(glucose, offset_mgdl, lower, upper)`:
  composition of the two for convenience.

### Dashboard Integration (Tab 3)

Tab 3 ("🔧 Counterfactual Metrics (Target Range & CGM Bias)") exposes two
user-controllable parameter groups:

- **Target range sliders**: `cf_lower`, `cf_upper` (defaults 70, 180 mg/dL).
- **CGM bias slider**: `cgm_offset` in [−40, 40] mg/dL.

For the currently selected subject and date window (`df_filtered`), the
application computes three scenarios:

1. Observed metrics under `[70, 180]` (canonical ADA-style range).
2. Counterfactual metrics under `[cf_lower, cf_upper]` with the observed
   CGM trace.
3. Counterfactual metrics under `[cf_lower, cf_upper]` after applying the
   uniform bias `cgm_offset`.

All three scenarios are displayed side-by-side in a compact table and as an
overlay of the observed glucose trace and its biased version, with the
counterfactual target band rendered as a shaded region.

This design keeps the mathematics transparent and directly tied to the
published TIR definitions, while leaving room for future, model-based
therapy counterfactuals.

## File Locations

### Key Files
- `app.py`: Streamlit dashboard entry point
- `src/data_loader.py`: Core data loading logic (well-documented with PhD-level docstrings)
- `tests/test_data_loader.py`: Comprehensive test suite
- `requirements.txt`: All dependencies (includes streamlit, pandas, plotly, pytest, dvc)
- `data/raw.dvc`: DVC tracking file for data directory

### Documentation
- `README.md`: Project overview and usage instructions
- `VERIFICATION_REPORT.md`: Test results and data quality report
- `LICENSE`: Apache 2.0

### Virtual Environment
Always use `.venv-azt1d/` as the virtual environment name (already in `.gitignore`).

## Additional Notes

- This is a research project by ASU Students at Arizona State University
- Data represents real-world Type 1 Diabetes patients on AID systems (Dec 2023 - Jan 2024)
- Licensed under Apache 2.0
- Version: 0.0.4 (early development stage)