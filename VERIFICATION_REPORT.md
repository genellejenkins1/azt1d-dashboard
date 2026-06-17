# AZT1D Dashboard - System Verification Report

**Authors:** Naif A. Ganadily, Genelle Jenkins, Toshika Talele  
**Lab:** Prof Hassan and PhD Student Saman, Arizona State University 
**Date:** October 23, 2025  
**Version:** 0.0.4

---

## Executive Summary

This document reports on comprehensive testing of the AZT1D Dashboard data infrastructure and visualization system. We validated 49 unit tests, data loading for all 25 subjects, and Streamlit dashboard functionality. After correcting a schema inconsistency in Subject 14, all subjects now load successfully with 100% coverage.

---

## Testing Methodology

### Unit Test Suite
We executed the complete test battery using pytest, covering:
- Initialization and configuration validation
- Schema enforcement mechanisms
- Single and batch subject loading protocols
- Data preprocessing pipelines
- Derived feature computation (glucose zones, clinical indicators)
- Error handling and edge cases

**Results:** 49/49 core tests passed. Two performance benchmarks were skipped due to missing `pytest-benchmark` dependency, which is optional for our use case.

### Subject-Level Data Validation
We systematically attempted to load all 25 subjects in the dataset to assess:
- File accessibility and format compliance
- CGM data completeness
- Schema validation
- Preprocessing robustness

Each subject was evaluated for both successful loading and CGM coverage percentage.

---

## Findings

### Successful Data Loading
All 25 subjects (100%) now load without errors. All subjects demonstrate:
- 100% CGM data coverage (no missing values post-preprocessing)
- Record counts ranging from 8,139 to 15,825 (mean ≈ 12,500)
- Proper datetime indexing and monotonic time series
- Valid derived clinical features

Representative examples:
- Subject 1: 11,042 records, mean glucose 142.1 mg/dL, 84.1% TIR
- Subject 14: 13,003 records, mean glucose 107.5 mg/dL, 92.6% TIR (now fixed)
- Subject 15: 15,825 records (highest count)
- Subject 8: 8,139 records (lowest count, still adequate for analysis)

### Subject 14 Schema Fix (RESOLVED)

**Issue Description:**  
Subject 14's CSV file originally used `Readings (CGM / BGM)` as the column header instead of the expected `CGM` identifier used by all other subjects. This caused schema validation to fail during the loading process.

**Technical Details:**
```
Expected schema:  EventDateTime, DeviceMode, BolusType, Basal, 
                  CorrectionDelivered, TotalBolusInsulinDelivered, 
                  FoodDelivered, CarbSize, CGM

Subject 14 schema: EventDateTime, Basal, BolusType, CarbSize, 
                   CorrectionDelivered, FoodDelivered, 
                   Readings (CGM / BGM), TotalBolusInsulinDelivered, 
                   DeviceMode
```

**Resolution Applied:**
We implemented a data correction script (`scripts/fix_subject14.py`) that:
1. Created a backup of the original file
2. Renamed the column from `Readings (CGM / BGM)` to `CGM`
3. Performed comprehensive data quality checks:
   - CGM data present: 13,003/13,003 records (100%)
   - CGM values in valid physiological range: 40-237 mg/dL (mean: 107.5)
   - All timestamps valid
   - All required columns present
   - Record count within expected range
4. Verified successful loading with the data loader

**Results:**
- Subject 14 now loads successfully
- 100% CGM data coverage maintained
- Mean glucose: 107.5 mg/dL
- Time in Range: 92.6% (excellent glycemic control)
- Backup preserved at: `data/raw/CGM Records/Subject 14/Subject 14.csv.backup`

---

## Dependency Verification

All required packages are installed and functional:

| Package | Version | Status |
|---------|---------|--------|
| pandas | 2.0.3 | ✓ |
| numpy | 1.24.4 | ✓ |
| streamlit | 1.40.1 | ✓ |
| plotly | 6.3.1 | ✓ |
| pytest | 8.3.5 | ✓ |
| pytest-cov | 5.0.0 | ✓ |

Requirements file has been updated to uncomment Streamlit and Plotly dependencies.

---

## Dashboard Functional Testing

### Operational Components
The Streamlit interface successfully instantiates with the following verified features:

**Data Selection & Filtering:**
- Subject dropdown populated with all 25 subjects
- Date range selector with validation
- Proper caching via `@st.cache_data` and `@st.cache_resource`

**Visualization (Tab 1):**
- Dual-panel Plotly figure with CGM time series (top) and insulin/carb events (bottom)
- ADA-aligned clinical thresholds rendered (70, 180 mg/dL)
- Target range shading (70-180 mg/dL, green overlay at 10% opacity)
- Basal insulin area chart with proper fill
- Bolus events as scatter markers
- Meal events (carbohydrates) on secondary y-axis
- Interactive hover tooltips with proper formatting

**Clinical Metrics (Tab 2):**
- 4x2 grid displaying 8 primary metrics
- Detailed statistics table (12 metrics)
- Proper percentage calculations for Time in Range (TIR), Time Below Range (TBR), Time Above Range (TAR)
- Coefficient of variation (CV) computed correctly

**Parameter Adjustment (Tab 3):**
- Placeholder content indicating future development
- Non-functional demo controls present as expected

### Performance Considerations
Caching mechanisms effectively prevent redundant data loading. Initial subject load takes approximately 1-2 seconds, subsequent access is near-instantaneous.

---

## How to Run

### Environment Activation
```bash
cd ~/Desktop/az_dashboard
source .venv-azt1d/bin/activate
```

### Execute Unit Tests
```bash
pytest tests/ -v -k "not performance"
```

### Launch Dashboard
```bash
streamlit run app.py
```

Dashboard accessible at `http://localhost:8501` in default browser.

---

## Complete Subject Loading Results

```
Subject  1: PASS (11,042 records, 100.0% CGM)
Subject  2: PASS (11,194 records, 100.0% CGM)
Subject  3: PASS (11,605 records, 100.0% CGM)
Subject  4: PASS (12,822 records, 100.0% CGM)
Subject  5: PASS (13,210 records, 100.0% CGM)
Subject  6: PASS (12,901 records, 100.0% CGM)
Subject  7: PASS (13,392 records, 100.0% CGM)
Subject  8: PASS (8,139 records, 100.0% CGM)
Subject  9: PASS (10,717 records, 100.0% CGM)
Subject 10: PASS (12,183 records, 100.0% CGM)
Subject 11: PASS (12,954 records, 100.0% CGM)
Subject 12: PASS (12,681 records, 100.0% CGM)
Subject 13: PASS (12,895 records, 100.0% CGM)
Subject 14: PASS (13,003 records, 100.0% CGM) ✓ FIXED
Subject 15: PASS (15,825 records, 100.0% CGM)
Subject 16: PASS (12,385 records, 100.0% CGM)
Subject 17: PASS (12,972 records, 100.0% CGM)
Subject 18: PASS (10,577 records, 100.0% CGM)
Subject 19: PASS (13,222 records, 100.0% CGM)
Subject 20: PASS (13,390 records, 100.0% CGM)
Subject 21: PASS (13,404 records, 100.0% CGM)
Subject 22: PASS (11,363 records, 100.0% CGM)
Subject 23: PASS (13,766 records, 100.0% CGM)
Subject 24: PASS (10,150 records, 100.0% CGM)
Subject 25: PASS (10,920 records, 100.0% CGM)
```

**Summary:** 25/25 subjects successful - 100% coverage achieved

---

## Next Steps

### Completed
1. ✓ Corrected Subject 14 column naming 
2. ✓ Re-validated all 25 subjects successfully
3. ✓ Implemented data quality checks in correction script

### Optional Improvements
1. Add pytest.ini to register custom marks (integration, performance) to suppress warnings
2. Consider adding automated data quality checks as pre-commit hooks

### Development Priorities
1. Implement counterfactual analysis module (Tab 3 functionality)
2. Add multi-subject comparison views
3. Integrate predictive modeling components (scikit-learn)
4. Develop export functionality for analysis results
5. Consider adding data quality metrics to dashboard

### Code Maintenance
- All 49 tests should continue passing with each modification
- Follow existing patterns for new feature development

---

## Conclusion

The AZT1D Dashboard infrastructure is robust and ready for research use. Data loading, preprocessing, and visualization components all function as designed. All 25 subjects now load successfully after correcting the Subject 14 schema issue. System performs efficiently with proper caching and handles the full dataset scale (~300K records across all subjects).

This verification confirms the platform is suitable for:
- Exploratory data analysis of T1D CGM data
- Clinical metric computation and visualization
- Foundation for counterfactual modeling development

The testing framework provides strong confidence in system reliability and will support continued development of advanced analytical features.

---

**Verification completed:** October 23, 2025  
**Status:** All 25 subjects operational - ready for counterfactual module development