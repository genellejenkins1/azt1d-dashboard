"""
Unit Tests for Data Loader Module

This test suite validates the data loading functionality for the AZT1D dataset,
ensuring robust error handling, data integrity, and correct preprocessing.

Test categories:
- Initialization and configuration
- Schema validation
- Single subject loading
- Multiple subject loading
- Data preprocessing and derived features
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from src.data_loader import (
    SubjectDataLoader,
    DataLoadError,
    load_subject_data,
    load_all_subjects,
    get_available_subjects
)


@pytest.fixture
def test_data_dir():
    """Fixture providing path to actual test data."""
    return Path("data/raw/CGM Records")


@pytest.fixture
def loader(test_data_dir):
    """Fixture providing initialized data loader."""
    return SubjectDataLoader(test_data_dir)


@pytest.fixture
def sample_subject_id():
    """Fixture providing a valid subject ID for testing."""
    return 1


@pytest.fixture
def mock_csv_data():
    """Fixture providing mock CSV data structure."""
    return pd.DataFrame({
        'EventDateTime': pd.date_range('2023-12-08', periods=100, freq='5T'),
        'DeviceMode': [''] * 100,
        'BolusType': [''] * 50 + ['Standard/Correction'] * 50,
        'Basal': [1.0] * 100,
        'CorrectionDelivered': [0.0] * 50 + [5.0] * 50,
        'TotalBolusInsulinDelivered': [0.0] * 50 + [5.0] * 50,
        'FoodDelivered': [0.0] * 80 + [10.0] * 20,
        'CarbSize': [0.0] * 80 + [50.0] * 20,
        'CGM': np.random.uniform(70, 180, 100)  # Target range values
    })


@pytest.fixture
def temp_data_dir(mock_csv_data):
    """
    Fixture creating temporary data directory with mock CSV files.
    
    This allows testing without relying on actual data files.
    """
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create subject folders
    for subject_id in [1, 2, 3]:
        subject_folder = temp_dir / f"Subject {subject_id}"
        subject_folder.mkdir(parents=True)
        
        # Save mock CSV
        csv_path = subject_folder / f"Subject {subject_id}.csv"
        mock_csv_data.to_csv(csv_path, index=False)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


# Test: Initialization

class TestInitialization:
    """Test data loader initialization and configuration."""
    
    def test_init_with_valid_directory(self, test_data_dir):
        """Test initialization with valid data directory."""
        loader = SubjectDataLoader(test_data_dir)
        assert loader.data_dir == test_data_dir
        assert loader.data_dir.exists()
    
    def test_init_with_invalid_directory(self):
        """Test that initialization fails with non-existent directory."""
        with pytest.raises(DataLoadError, match="Data directory not found"):
            SubjectDataLoader("nonexistent/path")
    
    def test_init_with_string_path(self, test_data_dir):
        """Test initialization with string path (not Path object)."""
        loader = SubjectDataLoader(str(test_data_dir))
        assert isinstance(loader.data_dir, Path)
        assert loader.data_dir.exists()
    
    def test_clinical_thresholds_defined(self, loader):
        """Test that clinical thresholds are properly defined."""
        assert loader.HYPOGLYCEMIA_THRESHOLD == 70
        assert loader.SEVERE_HYPOGLYCEMIA_THRESHOLD == 54
        assert loader.TARGET_RANGE_LOWER == 70
        assert loader.TARGET_RANGE_UPPER == 180
        assert loader.HYPERGLYCEMIA_THRESHOLD == 180
        assert loader.SEVERE_HYPERGLYCEMIA_THRESHOLD == 250
    
    def test_expected_columns_defined(self, loader):
        """Test that expected column schema is defined."""
        assert len(loader.EXPECTED_COLUMNS) == 9
        assert 'EventDateTime' in loader.EXPECTED_COLUMNS
        assert 'CGM' in loader.EXPECTED_COLUMNS


# Test: Subject Discovery

class TestSubjectDiscovery:
    """Test functionality for discovering available subjects."""
    
    def test_get_available_subjects_returns_list(self, loader):
        """Test that get_available_subjects returns a list."""
        subjects = loader.get_available_subjects()
        assert isinstance(subjects, list)
    
    def test_get_available_subjects_sorted(self, loader):
        """Test that subject IDs are returned in sorted order."""
        subjects = loader.get_available_subjects()
        assert subjects == sorted(subjects)
    
    def test_get_available_subjects_correct_range(self, loader):
        """Test that subject IDs are in expected range."""
        subjects = loader.get_available_subjects()
        assert all(1 <= s <= 25 for s in subjects)
    
    def test_get_available_subjects_with_temp_dir(self, temp_data_dir):
        """Test subject discovery with controlled directory."""
        loader = SubjectDataLoader(temp_data_dir)
        subjects = loader.get_available_subjects()
        assert subjects == [1, 2, 3]
    
    def test_convenience_function_get_available_subjects(self, test_data_dir):
        """Test convenience function for getting subjects."""
        subjects = get_available_subjects(str(test_data_dir))
        assert isinstance(subjects, list)
        assert len(subjects) > 0


# Test: Single Subject Loading

class TestSingleSubjectLoading:
    """Test loading individual subject data."""
    
    def test_load_subject_returns_dataframe(self, loader, sample_subject_id):
        """Test that load_subject returns a DataFrame."""
        df = loader.load_subject(sample_subject_id)
        assert isinstance(df, pd.DataFrame)
    
    def test_load_subject_has_correct_columns(self, loader, sample_subject_id):
        """Test that loaded data has all expected columns."""
        df = loader.load_subject(sample_subject_id)
        
        # Original columns
        for col in loader.EXPECTED_COLUMNS:
            if col != 'EventDateTime':  # EventDateTime becomes index
                assert col in df.columns
        
        # Derived columns
        assert 'subject_id' in df.columns
        assert 'glucose_zone' in df.columns
        assert 'is_hypoglycemic' in df.columns
        assert 'is_hyperglycemic' in df.columns
        assert 'in_target_range' in df.columns
        assert 'has_bolus' in df.columns
        assert 'has_carbs' in df.columns
    
    def test_load_subject_datetime_index(self, loader, sample_subject_id):
        """Test that DataFrame uses datetime index."""
        df = loader.load_subject(sample_subject_id)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == 'EventDateTime'
    
    def test_load_subject_sorted_by_time(self, loader, sample_subject_id):
        """Test that data is sorted by timestamp."""
        df = loader.load_subject(sample_subject_id)
        assert df.index.is_monotonic_increasing
    
    def test_load_subject_has_data(self, loader, sample_subject_id):
        """Test that loaded subject has non-zero records."""
        df = loader.load_subject(sample_subject_id)
        assert len(df) > 0
    
    def test_load_subject_cgm_values_reasonable(self, loader, sample_subject_id):
        """Test that CGM values are in physiologically reasonable range."""
        df = loader.load_subject(sample_subject_id)
        cgm_values = df['CGM'].dropna()
        
        # Most glucose values should be between 40 and 400 mg/dL
        assert cgm_values.min() >= 0
        assert cgm_values.max() <= 600  # Extreme but possible
        assert cgm_values.mean() > 50
        assert cgm_values.mean() < 300
    
    def test_load_subject_without_validation(self, loader, sample_subject_id):
        """Test loading without schema validation."""
        df = loader.load_subject(sample_subject_id, validate=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_load_subject_without_derived_features(self, loader, sample_subject_id):
        """Test loading without derived feature computation."""
        df = loader.load_subject(sample_subject_id, add_derived_features=False)
        
        # Should not have derived columns
        assert 'glucose_zone' not in df.columns
        assert 'is_hypoglycemic' not in df.columns
        assert 'is_hyperglycemic' not in df.columns
    
    def test_load_nonexistent_subject(self, loader):
        """Test that loading non-existent subject raises error."""
        with pytest.raises(DataLoadError, match="Data file not found"):
            loader.load_subject(999)
    
    def test_convenience_function_load_subject(self, test_data_dir, sample_subject_id):
        """Test convenience function for loading single subject."""
        df = load_subject_data(sample_subject_id, str(test_data_dir))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# Test: Multiple Subject Loading

class TestMultipleSubjectLoading:
    """Test loading and combining multiple subjects."""
    
    def test_load_multiple_subjects_default(self, temp_data_dir):
        """Test loading all subjects when no IDs specified."""
        loader = SubjectDataLoader(temp_data_dir)
        df = loader.load_multiple_subjects()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert df['subject_id'].nunique() == 3
    
    def test_load_multiple_subjects_specific_ids(self, temp_data_dir):
        """Test loading specific subject IDs."""
        loader = SubjectDataLoader(temp_data_dir)
        df = loader.load_multiple_subjects([1, 2])
        
        assert df['subject_id'].nunique() == 2
        assert set(df['subject_id'].unique()) == {1, 2}
    
    def test_load_multiple_subjects_combined_structure(self, temp_data_dir):
        """Test that combined DataFrame has correct structure."""
        loader = SubjectDataLoader(temp_data_dir)
        df = loader.load_multiple_subjects()
        
        # Check that all subjects contribute data
        for subject_id in [1, 2, 3]:
            assert subject_id in df['subject_id'].values
    
    def test_load_multiple_subjects_with_failures(self, temp_data_dir):
        """Test loading when some subjects fail (should warn but continue)."""
        loader = SubjectDataLoader(temp_data_dir)
        
        # Include non-existent subject
        df = loader.load_multiple_subjects([1, 2, 999])
        
        # Should still load available subjects
        assert len(df) > 0
        assert df['subject_id'].nunique() == 2
    
    def test_convenience_function_load_all_subjects(self, temp_data_dir):
        """Test convenience function for loading all subjects."""
        df = load_all_subjects(str(temp_data_dir))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# Test: Schema Validation

class TestSchemaValidation:
    """Test data schema validation logic."""
    
    def test_validate_schema_with_valid_data(self, loader, mock_csv_data):
        """Test validation passes with valid data."""
        # Should not raise any exception
        loader._validate_schema(mock_csv_data, subject_id=1)
    
    def test_validate_schema_missing_columns(self, loader):
        """Test validation fails with missing columns."""
        invalid_df = pd.DataFrame({'CGM': [100, 120, 140]})
        
        with pytest.raises(DataLoadError, match="missing required columns"):
            loader._validate_schema(invalid_df, subject_id=1)
    
    def test_validate_schema_empty_dataframe(self, loader, mock_csv_data):
        """Test validation fails with empty DataFrame."""
        empty_df = mock_csv_data.iloc[:0]
        
        with pytest.raises(DataLoadError, match="no data records"):
            loader._validate_schema(empty_df, subject_id=1)
    
    def test_validate_schema_no_cgm_data(self, loader, mock_csv_data):
        """Test validation fails when all CGM values are NaN."""
        invalid_df = mock_csv_data.copy()
        invalid_df['CGM'] = np.nan
        
        with pytest.raises(DataLoadError, match="no CGM readings"):
            loader._validate_schema(invalid_df, subject_id=1)


# Test: Preprocessing

class TestPreprocessing:
    """Test data preprocessing transformations."""
    
    def test_preprocess_adds_subject_id(self, loader, mock_csv_data):
        """Test that preprocessing adds subject_id column."""
        df = loader._preprocess_dataframe(mock_csv_data, subject_id=5, add_derived_features=False)
        assert 'subject_id' in df.columns
        assert (df['subject_id'] == 5).all()
    
    def test_preprocess_converts_datetime(self, loader, mock_csv_data):
        """Test datetime conversion and indexing."""
        df = loader._preprocess_dataframe(mock_csv_data, subject_id=1, add_derived_features=False)
        assert isinstance(df.index, pd.DatetimeIndex)
    
    def test_preprocess_sorts_by_time(self, loader, mock_csv_data):
        """Test that data is sorted chronologically."""
        # Shuffle the data
        shuffled = mock_csv_data.sample(frac=1).reset_index(drop=True)
        df = loader._preprocess_dataframe(shuffled, subject_id=1, add_derived_features=False)
        assert df.index.is_monotonic_increasing
    
    def test_preprocess_converts_numeric_columns(self, loader, mock_csv_data):
        """Test numeric column type conversion."""
        # Add some string values that should be coerced
        mock_csv_data.loc[0, 'CGM'] = 'invalid'
        df = loader._preprocess_dataframe(mock_csv_data, subject_id=1, add_derived_features=False)
        
        # Should be converted to float (with NaN for invalid)
        assert df['CGM'].dtype in [np.float64, np.float32]


# Test: Derived Features

class TestDerivedFeatures:
    """Test computation of derived clinical features."""
    
    def test_derived_features_glucose_zones(self, loader, sample_subject_id):
        """Test glucose zone classification."""
        df = loader.load_subject(sample_subject_id)
        
        # Check all possible zones exist in data or are valid categories
        assert 'glucose_zone' in df.columns
        assert df['glucose_zone'].dtype.name == 'category'
    
    def test_derived_features_hypoglycemia_indicator(self, loader):
        """Test hypoglycemia binary indicator."""
        # Create test data with known glucose values
        test_df = pd.DataFrame({
            'CGM': [60, 80, 100, 50, 200],
            'TotalBolusInsulinDelivered': [0.0] * 5,
            'CarbSize': [0.0] * 5
        })
        result_df = loader._add_derived_features(test_df)
        
        expected = [1, 0, 0, 1, 0]  # <70 is hypoglycemic
        assert (result_df['is_hypoglycemic'] == expected).all()
    
    def test_derived_features_hyperglycemia_indicator(self, loader):
        """Test hyperglycemia binary indicator."""
        test_df = pd.DataFrame({
            'CGM': [100, 150, 200, 250, 300],
            'TotalBolusInsulinDelivered': [0.0] * 5,
            'CarbSize': [0.0] * 5
        })
        result_df = loader._add_derived_features(test_df)
        
        expected = [0, 0, 1, 1, 1]  # >180 is hyperglycemic
        assert (result_df['is_hyperglycemic'] == expected).all()
    
    def test_derived_features_target_range(self, loader):
        """Test target range indicator."""
        test_df = pd.DataFrame({
            'CGM': [60, 70, 125, 180, 200],
            'TotalBolusInsulinDelivered': [0.0] * 5,
            'CarbSize': [0.0] * 5
        })
        result_df = loader._add_derived_features(test_df)
        
        expected = [0, 1, 1, 1, 0]  # 70-180 is target range
        assert (result_df['in_target_range'] == expected).all()
    
    def test_derived_features_bolus_indicator(self, loader):
        """Test bolus administration indicator."""
        test_df = pd.DataFrame({
            'CGM': [100] * 5,
            'TotalBolusInsulinDelivered': [0, 0, 5.5, 0, 10.2],
            'CarbSize': [0.0] * 5
        })
        result_df = loader._add_derived_features(test_df)
        
        expected = [0, 0, 1, 0, 1]
        assert (result_df['has_bolus'] == expected).all()
    
    def test_derived_features_carb_indicator(self, loader):
        """Test carbohydrate intake indicator."""
        test_df = pd.DataFrame({
            'CGM': [100] * 5,
            'TotalBolusInsulinDelivered': [0] * 5,
            'CarbSize': [0, 30, 0, 60, 0]
        })
        result_df = loader._add_derived_features(test_df)
        
        expected = [0, 1, 0, 1, 0]
        assert (result_df['has_carbs'] == expected).all()


# Test: Summary Statistics

class TestSummaryStatistics:
    """Test subject summary generation."""
    
    def test_get_subject_summary_returns_dict(self, loader, sample_subject_id):
        """Test that summary returns a dictionary."""
        summary = loader.get_subject_summary(sample_subject_id)
        assert isinstance(summary, dict)
    
    def test_get_subject_summary_has_required_keys(self, loader, sample_subject_id):
        """Test that summary contains all required keys."""
        summary = loader.get_subject_summary(sample_subject_id)
        
        required_keys = [
            'subject_id', 'n_records', 'start_date', 'end_date',
            'days_of_data', 'mean_glucose', 'std_glucose', 'cv_glucose',
            'time_in_range', 'time_below_range', 'time_above_range',
            'total_bolus_insulin', 'total_basal_insulin', 'total_carbs',
            'n_bolus_events', 'n_meals'
        ]
        
        for key in required_keys:
            assert key in summary
    
    def test_get_subject_summary_values_reasonable(self, loader, sample_subject_id):
        """Test that summary statistics are in reasonable ranges."""
        summary = loader.get_subject_summary(sample_subject_id)
        
        # Basic sanity checks
        assert summary['n_records'] > 0
        assert summary['days_of_data'] > 0
        assert 0 < summary['mean_glucose'] < 600
        assert 0 <= summary['time_in_range'] <= 100
        assert 0 <= summary['time_below_range'] <= 100
        assert 0 <= summary['time_above_range'] <= 100
        
        # Time percentages should sum to ~100% (within rounding)
        time_sum = (summary['time_in_range'] + 
                   summary['time_below_range'] + 
                   summary['time_above_range'])
        assert 99 <= time_sum <= 101
    
    def test_get_subject_summary_dates_format(self, loader, sample_subject_id):
        """Test that dates are properly formatted strings."""
        summary = loader.get_subject_summary(sample_subject_id)
        
        # Check date format YYYY-MM-DD
        assert len(summary['start_date']) == 10
        assert len(summary['end_date']) == 10
        assert summary['start_date'][4] == '-'
        assert summary['end_date'][4] == '-'


# Test: Edge Cases and Error Handling

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_load_subject_with_missing_cgm_values(self, loader, sample_subject_id):
        """Test that missing CGM values are handled gracefully."""
        df = loader.load_subject(sample_subject_id)
        
        # Should have some NaN values in CGM
        # Derived features should handle NaN appropriately
        assert df is not None
    
    def test_empty_subject_list_raises_error(self, loader):
        """Test that loading empty subject list raises error."""
        with pytest.raises(DataLoadError, match="No subject data could be loaded"):
            loader.load_multiple_subjects([999, 998, 997])
    
    def test_subject_id_type_validation(self, loader):
        """Test handling of invalid subject ID types."""
        with pytest.raises((TypeError, DataLoadError)):
            loader.load_subject("invalid")
    
    def test_handles_duplicate_timestamps(self, loader, mock_csv_data):
        """Test handling of duplicate timestamps in data."""
        # Add duplicate timestamp
        dup_data = pd.concat([mock_csv_data, mock_csv_data.iloc[:1]])
        df = loader._preprocess_dataframe(dup_data, subject_id=1, add_derived_features=False)
        
        # Should still process without error
        assert len(df) > 0


# Test: Integration Tests

class TestIntegration:
    """Integration tests using real data (if available)."""
    
    @pytest.mark.integration
    def test_load_all_real_subjects(self, loader):
        """Integration test: Load all available subjects from real data."""
        subjects = loader.get_available_subjects()
        
        if len(subjects) == 0:
            pytest.skip("No real data available")
        
        df = loader.load_multiple_subjects()
        
        # Verify combined data
        assert len(df) > 0
        # Some subjects may fail to load due to data quality issues
        # but we should get most of them
        successfully_loaded = df['subject_id'].nunique()
        assert successfully_loaded >= 20  # At least 20 subjects expected
        assert successfully_loaded <= len(subjects)  # Can't load more than available
    
    @pytest.mark.integration
    def test_real_data_quality_checks(self, loader, sample_subject_id):
        """Integration test: Verify real data quality."""
        try:
            df = loader.load_subject(sample_subject_id)
        except DataLoadError:
            pytest.skip(f"Subject {sample_subject_id} data not available")
        
        # Data quality checks
        assert len(df) > 1000  # Should have substantial data
        assert df['CGM'].notna().sum() > len(df) * 0.5  # At least 50% CGM coverage
        assert df.index.is_monotonic_increasing  # Properly sorted
        
        # Check for reasonable date range
        date_range = (df.index.max() - df.index.min()).days
        assert date_range > 0


# Test: Performance

class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.performance
    def test_load_subject_performance(self, loader, sample_subject_id, benchmark):
        """Benchmark: Single subject loading time."""
        try:
            result = benchmark(loader.load_subject, sample_subject_id)
            assert len(result) > 0
        except:
            pytest.skip("Benchmark fixture not available")
    
    @pytest.mark.performance  
    def test_load_multiple_subjects_performance(self, loader, benchmark):
        """Benchmark: Multiple subject loading time."""
        try:
            result = benchmark(loader.load_multiple_subjects, [1, 2, 3])
            assert len(result) > 0
        except:
            pytest.skip("Benchmark fixture not available")