"""
Data Loader Module for AZT1D 2025 Dataset

This module provides robust data loading and preprocessing functionality for the
AZT1D (Arizona Type 1 Diabetes) 2025 dataset, which contains real-world CGM and
insulin delivery data from 25 individuals using Automated Insulin Delivery (AID) systems.

The data includes:
- Continuous Glucose Monitoring (CGM) readings at 5-minute intervals
- Insulin delivery data (bolus and basal rates)
- Carbohydrate intake records
- Device operating modes (normal, sleep, exercise)

Author: Naif A. Ganadily, Genelle Jenkins, Toshika Talele
Date: October 2025
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Union
import warnings

import pandas as pd
import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Exception for data loading errors."""
    pass


class SubjectDataLoader:
    """
    Loader for individual subject CGM and insulin delivery data.
    
    This class handles loading, validation, and preprocessing of time-series
    data for Type 1 Diabetes research.
    
    Attributes:
        data_dir (Path): Root directory containing subject data folders.
        expected_columns (List[str]): Expected column names in CSV files.
    """
    
    # Expected schema for validation and testing
    EXPECTED_COLUMNS = [
        'EventDateTime', 'DeviceMode', 'BolusType', 'Basal',
        'CorrectionDelivered', 'TotalBolusInsulinDelivered',
        'FoodDelivered', 'CarbSize', 'CGM'
    ]
    
    # Clinical thresholds (mg/dL)
    HYPOGLYCEMIA_THRESHOLD = 70   # Level 1 hypoglycemia
    SEVERE_HYPOGLYCEMIA_THRESHOLD = 54  # Level 2 hypoglycemia
    TARGET_RANGE_LOWER = 70
    TARGET_RANGE_UPPER = 180
    HYPERGLYCEMIA_THRESHOLD = 180  # Level 1 hyperglycemia
    SEVERE_HYPERGLYCEMIA_THRESHOLD = 250  # Level 2 hyperglycemia
    
    def __init__(self, data_dir: Union[str, Path] = "data/raw/CGM Records"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to directory containing subject folders.
                     Default is "data/raw/CGM Records".
        
        Raises:
            DataLoadError: If the data directory does not exist.
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise DataLoadError(
                f"Data directory not found: {self.data_dir}\n"
                "Please ensure DVC data is pulled: dvc pull command"
            )
        
        logger.info(f"Initialized DataLoader with directory: {self.data_dir}")
    
    def get_available_subjects(self) -> List[int]:
        """
        Get list of subject IDs with available data.
        
        Returns:
            Sorted list of subject IDs (integers).
        
        Example:
            >>> loader = SubjectDataLoader()
            >>> subjects = loader.get_available_subjects()
            >>> print(subjects)  #i.e. [1, 2, 3, ..., 25]
        """
        subjects = []
        
        for subject_dir in self.data_dir.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith("Subject"):
                try:
                    subject_id = int(subject_dir.name.split()[-1])
                    subjects.append(subject_id)
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse subject ID from: {subject_dir.name}")
        
        return sorted(subjects)
    
    def load_subject(self, subject_id: int, validate: bool = True, add_derived_features: bool = True) -> pd.DataFrame:
        """
        Load data for a single subject with validation and preprocessing.
        
        Args:
            subject_id: Subject ID number (1-25).
            validate: Whether to validate data schema and quality.
            add_derived_features: Whether to add derived clinical features.
        
        Returns:
            DataFrame with subject's CGM and insulin data, indexed by datetime.
        
        Raises:
            DataLoadError: If subject data cannot be loaded or is invalid.
        
        Example:
            >>> loader = SubjectDataLoader()
            >>> df = loader.load_subject(1)
            >>> print(df.shape)  # (11042, 13)
            >>> print(df['CGM'].mean())  # Average glucose level
        """
        # Construct file path
        subject_folder = self.data_dir / f"Subject {subject_id}"
        csv_file = subject_folder / f"Subject {subject_id}.csv"
        
        if not csv_file.exists():
            raise DataLoadError(
                f"Data file not found for Subject {subject_id}: {csv_file}"
            )
        
        # Load CSV
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} records for Subject {subject_id}")
        except Exception as e:
            raise DataLoadError(
                f"Failed to read CSV for Subject {subject_id}: {str(e)}"
            )
        
        # Validate schema
        if validate:
            self._validate_schema(df, subject_id)
        
        # Preprocess
        df = self._preprocess_dataframe(df, subject_id, add_derived_features)
        
        return df
    
    def _validate_schema(self, df: pd.DataFrame, subject_id: int) -> None:
        """
        Validate that DataFrame has expected columns and basic data quality.
        
        Args:
            df: DataFrame to validate.
            subject_id: Subject ID for error messages.
        
        Raises:
            DataLoadError: If validation fails.
        """
        # Check columns
        missing_cols = set(self.EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise DataLoadError(
                f"Subject {subject_id} missing required columns: {missing_cols}"
            )
        
        # Check for completely empty DataFrame
        if df.empty:
            raise DataLoadError(f"Subject {subject_id} has no data records")
        
        # Check CGM column has some data
        if df['CGM'].isna().all():
            raise DataLoadError(
                f"Subject {subject_id} has no CGM readings"
            )
        
        logger.debug(f"Schema validation passed for Subject {subject_id}")
    
    def _preprocess_dataframe(self, df: pd.DataFrame, subject_id: int,add_derived_features: bool) -> pd.DataFrame:
        """
        Apply preprocessing transformations to raw data.
        
        Args:
            df: Raw DataFrame from CSV.
            subject_id: Subject ID for tracking.
            add_derived_features: Whether to compute derived features.
        
        Returns:
            Preprocessed DataFrame with standardized types and features.
        """
        df = df.copy()
        
        # Add subject identifier
        df['subject_id'] = subject_id
        
        # Parse datetime and set as index
        df['EventDateTime'] = pd.to_datetime(df['EventDateTime'], errors='coerce')
        
        # Check for datetime parsing errors
        n_invalid_dates = df['EventDateTime'].isna().sum()
        if n_invalid_dates > 0:
            warnings.warn(
                f"Subject {subject_id}: {n_invalid_dates} records with invalid timestamps"
            )
        
        # Sort by datetime and set index
        df = df.sort_values('EventDateTime').set_index('EventDateTime')
        
        # Convert numeric columns to float
        numeric_cols = [
            'Basal', 'CorrectionDelivered', 'TotalBolusInsulinDelivered',
            'FoodDelivered', 'CarbSize', 'CGM'
        ]
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add derived clinical features
        if add_derived_features:
            df = self._add_derived_features(df)
        
        logger.debug(f"Preprocessing complete for Subject {subject_id}")
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived clinical features for analysis.
        
        These features are commonly used in diabetes research and clinical practice.
        
        Args:
            df: Preprocessed DataFrame.
        
        Returns:
            DataFrame with additional derived columns.
        """
        # Glucose zone classification
        # Bins: 0-54 (severe hypo), 54-70 (hypo), 70-180 (target range), 
        #       180-250 (hyper), 250+ (severe hyper)
        df['glucose_zone'] = pd.cut(
            df['CGM'],
            bins=[0, self.SEVERE_HYPOGLYCEMIA_THRESHOLD, 
                  self.HYPOGLYCEMIA_THRESHOLD, 
                  self.HYPERGLYCEMIA_THRESHOLD,
                  self.SEVERE_HYPERGLYCEMIA_THRESHOLD, float('inf')],
            labels=['severe_hypo', 'hypo', 'in_range', 'hyper', 'severe_hyper']
        )
        
        # Binary indicators for clinical events
        df['is_hypoglycemic'] = (df['CGM'] < self.HYPOGLYCEMIA_THRESHOLD).astype(int)
        df['is_hyperglycemic'] = (df['CGM'] > self.HYPERGLYCEMIA_THRESHOLD).astype(int)
        df['in_target_range'] = (
            (df['CGM'] >= self.TARGET_RANGE_LOWER) & 
            (df['CGM'] <= self.TARGET_RANGE_UPPER)
        ).astype(int)
        
        # Insulin administration indicators
        df['has_bolus'] = (df['TotalBolusInsulinDelivered'] > 0).astype(int)
        df['has_carbs'] = (df['CarbSize'] > 0).astype(int)
        
        return df
    
    def load_multiple_subjects(self, subject_ids: Optional[List[int]] = None, validate: bool = True, add_derived_features: bool = True) -> pd.DataFrame:
        """
        Load and combine data from multiple subjects.
        
        Args:
            subject_ids: List of subject IDs to load. If None, loads all available.
            validate: Whether to validate each subject's data.
            add_derived_features: Whether to add derived features.
        
        Returns:
            Combined DataFrame with all subjects' data.
        
        Example:
            >>> loader = SubjectDataLoader()
            >>> # Load all subjects
            >>> df_all = loader.load_multiple_subjects()
            >>> # Load specific subjects
            >>> df_subset = loader.load_multiple_subjects([1, 2, 3])
        """
        if subject_ids is None:
            subject_ids = self.get_available_subjects()
        
        dataframes = []
        failed_subjects = []
        
        for subject_id in subject_ids:
            try:
                df = self.load_subject(
                    subject_id, 
                    validate=validate,
                    add_derived_features=add_derived_features
                )
                dataframes.append(df)
            except DataLoadError as e:
                logger.warning(f"Failed to load Subject {subject_id}: {str(e)}")
                failed_subjects.append(subject_id)
        
        if not dataframes:
            raise DataLoadError("No subject data could be loaded")
        
        if failed_subjects:
            logger.warning(f"Failed to load {len(failed_subjects)} subjects: {failed_subjects}")
        
        # Combine all subjects
        combined_df = pd.concat(dataframes, axis=0)
        logger.info(
            f"Successfully loaded {len(dataframes)} subjects "
            f"with {len(combined_df)} total records"
        )
        
        return combined_df
    
    def get_subject_summary(self, subject_id: int) -> Dict[str, Union[int, float, str]]:
        """
        Generate summary statistics for a subject without loading full data.
        
        Args:
            subject_id: Subject ID to summarize.
        
        Returns:
            Dictionary with summary statistics.
        
        Example:
            >>> loader = SubjectDataLoader()
            >>> summary = loader.get_subject_summary(1)
            >>> print(summary['mean_glucose'])
            >>> print(summary['days_of_data'])
        """
        df = self.load_subject(subject_id)
        
        summary = {
            'subject_id': subject_id,
            'n_records': len(df),
            'start_date': df.index.min().strftime('%Y-%m-%d'),
            'end_date': df.index.max().strftime('%Y-%m-%d'),
            'days_of_data': (df.index.max() - df.index.min()).days + 1,
            'mean_glucose': df['CGM'].mean(),
            'std_glucose': df['CGM'].std(),
            'cv_glucose': (df['CGM'].std() / df['CGM'].mean()) * 100,  # Coefficient of variation
            'time_in_range': (df['in_target_range'].sum() / len(df)) * 100,
            'time_below_range': (df['is_hypoglycemic'].sum() / len(df)) * 100,
            'time_above_range': (df['is_hyperglycemic'].sum() / len(df)) * 100,
            'total_bolus_insulin': df['TotalBolusInsulinDelivered'].sum(),
            'total_basal_insulin': df['Basal'].sum(),
            'total_carbs': df['CarbSize'].sum(),
            'n_bolus_events': df['has_bolus'].sum(),
            'n_meals': df['has_carbs'].sum(),
        }
        
        return summary


# Convenience functions for common use cases
def load_subject_data(subject_id: int, data_dir: str = "data/raw/CGM Records") -> pd.DataFrame:
    """
    Convenience function to load a single subject's data.
    
    Args:
        subject_id: Subject ID (1-25).
        data_dir: Path to data directory.
    
    Returns:
        DataFrame with subject's data.
    
    Example:
        >>> df = load_subject_data(1)
        >>> print(df.head())
    """
    loader = SubjectDataLoader(data_dir)
    return loader.load_subject(subject_id)


def load_all_subjects(data_dir: str = "data/raw/CGM Records") -> pd.DataFrame:
    """
    Convenience function to load all subjects' data.
    
    Args:
        data_dir: Path to data directory.
    
    Returns:
        Combined DataFrame with all subjects.
    
    Example:
        >>> df = load_all_subjects()
        >>> print(df.groupby('subject_id')['CGM'].mean())
    """
    loader = SubjectDataLoader(data_dir)
    return loader.load_multiple_subjects()


def get_available_subjects(data_dir: str = "data/raw/CGM Records") -> List[int]:
    """
    Convenience function to get list of available subjects.
    
    Args:
        data_dir: Path to data directory.
    
    Returns:
        List of subject IDs.
    
    Example:
        >>> subjects = get_available_subjects()
        >>> print(f"Found {len(subjects)} subjects")
    """
    loader = SubjectDataLoader(data_dir)
    return loader.get_available_subjects()
