#!/usr/bin/env python
"""
Data Correction Script: Subject 14 Schema Fix

This script corrects the column naming inconsistency in Subject 14's data file.
The column 'Readings (CGM / BGM)' is renamed to 'CGM' to match the standard schema.

A backup of the original file is created before modification.

Author: Naif A. Ganadily, Genelle Jenkins, Toshika Talele
Date: October 2025
"""

import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime

# File paths
DATA_DIR = Path("data/raw/CGM Records/Subject 14")
ORIGINAL_FILE = DATA_DIR / "Subject 14.csv"
BACKUP_FILE = DATA_DIR / "Subject 14.csv.backup"

def create_backup():
    """Create backup of original file"""
    print(f"Creating backup: {BACKUP_FILE}")
    shutil.copy2(ORIGINAL_FILE, BACKUP_FILE)
    print(f"✓ Backup created successfully")

def load_and_verify_original():
    """Load original file and verify the issue"""
    print(f"\nLoading original file: {ORIGINAL_FILE}")
    df = pd.read_csv(ORIGINAL_FILE)
    
    print(f"✓ Loaded {len(df)} records")
    print(f"\nOriginal columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # Verify the problematic column exists
    if 'Readings (CGM / BGM)' not in df.columns:
        raise ValueError("Expected column 'Readings (CGM / BGM)' not found!")
    
    if 'CGM' in df.columns:
        raise ValueError("Column 'CGM' already exists! No correction needed.")
    
    return df

def apply_correction(df):
    """Rename the problematic column to match standard schema."""
    print(f"\nApplying correction...")
    df_corrected = df.rename(columns={'Readings (CGM / BGM)': 'CGM'})
    print(f"✓ Renamed 'Readings (CGM / BGM)' -> 'CGM'")
    
    return df_corrected

def verify_correction(df):
    """Verify the correction was applied correctly."""
    print(f"\nVerifying correction...")
    
    # Check column exists
    if 'CGM' not in df.columns:
        raise ValueError("Correction failed: 'CGM' column not found!")
    
    # Check old column is gone
    if 'Readings (CGM / BGM)' in df.columns:
        raise ValueError("Correction failed: Old column still exists!")
    
    print(f"✓ Column 'CGM' exists")
    print(f"✓ Old column removed")
    
    # Display corrected columns
    print(f"\nCorrected columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    return True

def data_quality_check(df):
    """Perform data quality checks on corrected data."""
    print(f"\n{'='*60}")
    print(f"DATA QUALITY CHECKS")
    print(f"{'='*60}")
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: CGM data exists
    checks_total += 1
    cgm_count = df['CGM'].notna().sum()
    cgm_pct = cgm_count / len(df) * 100
    if cgm_count > 0:
        print(f"✓ CGM data present: {cgm_count}/{len(df)} records ({cgm_pct:.1f}%)")
        checks_passed += 1
    else:
        print(f"✗ No CGM data found!")
    
    # Check 2: CGM values in physiological range
    checks_total += 1
    cgm_values = df['CGM'].dropna()
    if len(cgm_values) > 0:
        min_cgm = cgm_values.min()
        max_cgm = cgm_values.max()
        mean_cgm = cgm_values.mean()
        
        if 0 <= min_cgm < 600 and max_cgm < 600 and 50 < mean_cgm < 300:
            print(f"✓ CGM values in valid range: {min_cgm:.1f} - {max_cgm:.1f} mg/dL (mean: {mean_cgm:.1f})")
            checks_passed += 1
        else:
            print(f"⚠ CGM values outside expected range: {min_cgm:.1f} - {max_cgm:.1f} mg/dL")
    else:
        print(f"✗ No CGM values to validate")
    
    # Check 3: DateTime column valid
    checks_total += 1
    try:
        df['EventDateTime'] = pd.to_datetime(df['EventDateTime'], errors='coerce')
        invalid_dates = df['EventDateTime'].isna().sum()
        if invalid_dates == 0:
            print(f"✓ All timestamps valid")
            checks_passed += 1
        else:
            print(f"⚠ {invalid_dates} invalid timestamps found")
    except Exception as e:
        print(f"✗ DateTime validation failed: {e}")
    
    # Check 4: Required columns present
    checks_total += 1
    required_cols = ['EventDateTime', 'DeviceMode', 'BolusType', 'Basal',
                    'CorrectionDelivered', 'TotalBolusInsulinDelivered',
                    'FoodDelivered', 'CarbSize', 'CGM']
    missing_cols = set(required_cols) - set(df.columns)
    if not missing_cols:
        print(f"✓ All required columns present")
        checks_passed += 1
    else:
        print(f"✗ Missing columns: {missing_cols}")
    
    # Check 5: No duplicate rows
    checks_total += 1
    n_duplicates = df.duplicated().sum()
    if n_duplicates == 0:
        print(f"✓ No duplicate rows")
        checks_passed += 1
    else:
        print(f"⚠ {n_duplicates} duplicate rows found")
    
    # Check 6: Record count reasonable
    checks_total += 1
    if 8000 <= len(df) <= 20000:
        print(f"✓ Record count in expected range: {len(df):,}")
        checks_passed += 1
    else:
        print(f"⚠ Record count outside typical range: {len(df):,}")
    
    print(f"\n{'='*60}")
    print(f"QUALITY SUMMARY: {checks_passed}/{checks_total} checks passed")
    print(f"{'='*60}")
    
    return checks_passed == checks_total

def save_corrected_file(df):
    """Save the corrected DataFrame to the original file."""
    print(f"\nSaving corrected file: {ORIGINAL_FILE}")
    df.to_csv(ORIGINAL_FILE, index=False)
    print(f"✓ File saved successfully")

def test_with_data_loader():
    """Test loading with the data loader to confirm fix works."""
    print(f"\n{'='*60}")
    print(f"TESTING WITH DATA LOADER")
    print(f"{'='*60}")
    
    try:
        from src.data_loader import SubjectDataLoader
        
        loader = SubjectDataLoader()
        df = loader.load_subject(14)
        
        print(f"✓ Subject 14 loaded successfully!")
        print(f"  Records: {len(df):,}")
        print(f"  CGM coverage: {df['CGM'].notna().sum()/len(df)*100:.1f}%")
        print(f"  Mean glucose: {df['CGM'].mean():.1f} mg/dL")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        return False

def main():
    """Main execution function."""
    print("="*60)
    print("Subject 14 Data Correction Script")
    print("="*60)
    
    try:
        # Step 1: Create backup
        create_backup()
        
        # Step 2: Load and verify original
        df_original = load_and_verify_original()
        
        # Step 3: Apply correction
        df_corrected = apply_correction(df_original)
        
        # Step 4: Verify correction
        verify_correction(df_corrected)
        
        # Step 5: Data quality checks
        quality_ok = data_quality_check(df_corrected)
        
        if not quality_ok:
            print("\n⚠ WARNING: Some quality checks failed. Review before saving.")
            response = input("Continue with save? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted. Original file unchanged.")
                return
        
        # Step 6: Save corrected file
        save_corrected_file(df_corrected)
        
        # Step 7: Test with data loader
        test_ok = test_with_data_loader()
        
        if test_ok:
            print("\n" + "="*60)
            print("SUCCESS! Subject 14 correction complete.")
            print("="*60)
            print(f"\nBackup saved to: {BACKUP_FILE}")
            print(f"To restore original: cp '{BACKUP_FILE}' '{ORIGINAL_FILE}'")
        else:
            print("\n⚠ WARNING: Correction saved but data loader test failed.")
            print("You may want to investigate further.")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("\nCorrection failed. Original file unchanged.")
        raise

if __name__ == "__main__":
    main()