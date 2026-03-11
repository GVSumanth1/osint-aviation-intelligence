"""
Flight Data Aggregation Script
===============================
Purpose: Process raw BTS flight data from Kaggle into daily aggregated statistics
Input:   flight_2024_data.csv (~1.2GB, 7M+ individual flight records)
Output:  flight_cancellations_daily_2024.csv (~20KB, 366 daily aggregations)

Author: Veekshith Sumanth (Team: Rishiv Bawa, Gaurav Sharma, Veekshith Sumanth)
Date: 2024
Project: Zero-Shot Classification for Flight Cancellation Prediction

Data Source:
- Kaggle: BTS On-Time Performance Dataset (2024)
- Original: Bureau of Transportation Statistics (https://www.transtats.bts.gov/)
- Format: Somewhat cleaned from raw BTS exports
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ==============================================================================
# STEP 1: LOAD RAW FLIGHT DATA
# ==============================================================================
print("=" * 80)
print("FLIGHT DATA AGGREGATION: BTS 2024 → DAILY STATISTICS")
print("=" * 80)
print("\nStep 1: Loading raw flight data from Kaggle (BTS source)...")

# Load the raw flight data
# This file contains individual flight records with cancellation status
df_flights = pd.read_csv('flight_2024_data.csv')

print(f"✓ Loaded {len(df_flights):,} flight records")
print(f"  Date range: {df_flights['fl_date'].min()} to {df_flights['fl_date'].max()}")
print(f"  Columns: {df_flights.columns.tolist()}")

# Display raw data sample
print("\nSample raw data:")
print(df_flights[['fl_date', 'origin', 'dest', 'cancelled', 'cancellation_code']].head(10))

# ==============================================================================
# STEP 2: DATA VALIDATION & CLEANING
# ==============================================================================
print("\n" + "=" * 80)
print("Step 2: Data Validation & Cleaning")
print("=" * 80)

# Convert fl_date to datetime
df_flights['fl_date'] = pd.to_datetime(df_flights['fl_date'])
print(f"✓ Converted fl_date to datetime format")

# Check for missing values in key columns
missing_dates = df_flights['fl_date'].isna().sum()
missing_cancelled = df_flights['cancelled'].isna().sum()

print(f"  Missing fl_date values: {missing_dates}")
print(f"  Missing cancelled values: {missing_cancelled}")

# Remove rows with missing critical data
if missing_dates > 0 or missing_cancelled > 0:
    df_flights = df_flights.dropna(subset=['fl_date', 'cancelled'])
    print(f"  Removed {missing_dates + missing_cancelled} rows with missing data")
    print(f"  Remaining records: {len(df_flights):,}")

# Validate cancelled column (should be 0 or 1)
unique_cancelled = df_flights['cancelled'].unique()
print(f"  Cancelled column unique values: {sorted(unique_cancelled)}")

# Ensure cancelled is binary (0 or 1)
df_flights['cancelled'] = df_flights['cancelled'].astype(int)

# ==============================================================================
# STEP 3: AGGREGATE TO DAILY LEVEL
# ==============================================================================
print("\n" + "=" * 80)
print("Step 3: Aggregating to Daily Level")
print("=" * 80)

# Group by date and calculate daily statistics
print("\nGrouping flights by date...")
daily_stats = df_flights.groupby('fl_date').agg({
    'cancelled': ['sum', 'count']  # sum = cancelled flights, count = total flights
}).reset_index()

# Flatten multi-level column names
daily_stats.columns = ['date', 'cancelled_flights', 'total_flights']

print(f"✓ Created daily aggregations")
print(f"  Total days: {len(daily_stats)}")
print(f"  Date range: {daily_stats['date'].min()} to {daily_stats['date'].max()}")

# ==============================================================================
# STEP 4: CALCULATE CANCELLATION RATE
# ==============================================================================
print("\n" + "=" * 80)
print("Step 4: Calculating Cancellation Rates")
print("=" * 80)

# Calculate cancellation rate as a percentage
daily_stats['cancellation_rate'] = (
    daily_stats['cancelled_flights'] / daily_stats['total_flights']
)

print(f"✓ Calculated daily cancellation rates")
print(f"\nCancellation Rate Statistics:")
print(f"  Mean: {daily_stats['cancellation_rate'].mean():.4f} ({daily_stats['cancellation_rate'].mean()*100:.2f}%)")
print(f"  Std Dev: {daily_stats['cancellation_rate'].std():.4f}")
print(f"  Min: {daily_stats['cancellation_rate'].min():.4f}")
print(f"  Max: {daily_stats['cancellation_rate'].max():.4f}")
print(f"  Median: {daily_stats['cancellation_rate'].median():.4f}")

# ==============================================================================
# STEP 5: IDENTIFY SPIKE DAYS (90TH PERCENTILE)
# ==============================================================================
print("\n" + "=" * 80)
print("Step 5: Identifying Cancellation Spike Days")
print("=" * 80)

# Calculate 90th percentile threshold for "high cancellation days"
threshold_90 = daily_stats['cancellation_rate'].quantile(0.90)
print(f"\n90th Percentile Threshold: {threshold_90:.4f} ({threshold_90*100:.2f}%)")

# Mark spike days
daily_stats['is_spike'] = daily_stats['cancellation_rate'] > threshold_90

# Count spike days
spike_count = daily_stats['is_spike'].sum()
spike_pct = (spike_count / len(daily_stats)) * 100

print(f"✓ Identified {spike_count} spike days ({spike_pct:.1f}% of all days)")

# Display spike days
print(f"\nSpike Days ({spike_count} total):")
spike_days = daily_stats[daily_stats['is_spike']].copy()
for idx, row in spike_days.iterrows():
    print(f"  {row['date'].date()}: {row['cancellation_rate']:.4f} "
          f"({row['cancelled_flights']:.0f}/{row['total_flights']:.0f} flights)")

# ==============================================================================
# STEP 6: VERIFY DATA QUALITY
# ==============================================================================
print("\n" + "=" * 80)
print("Step 6: Data Quality Verification")
print("=" * 80)

# Check for missing dates (should be 366 for leap year 2024)
expected_days = 366  # 2024 is a leap year
actual_days = len(daily_stats)

print(f"\nTemporal Coverage Check:")
print(f"  Expected days (2024 leap year): {expected_days}")
print(f"  Actual days in dataset: {actual_days}")

if actual_days == expected_days:
    print(f"  ✓ Complete coverage - no missing dates")
else:
    print(f"  ⚠ Warning: Missing {expected_days - actual_days} days")
    
    # Check for date gaps
    date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    missing_dates = set(date_range) - set(daily_stats['date'])
    if missing_dates:
        print(f"  Missing dates: {sorted(missing_dates)}")

# Validate no duplicate dates
duplicate_dates = daily_stats['date'].duplicated().sum()
print(f"\nDuplicate Check:")
print(f"  Duplicate dates: {duplicate_dates}")
if duplicate_dates == 0:
    print(f"  ✓ No duplicates found")

# Summary statistics
print(f"\nFinal Dataset Summary:")
print(f"  Total flights (sum): {daily_stats['total_flights'].sum():,}")
print(f"  Total cancelled (sum): {daily_stats['cancelled_flights'].sum():,}")
print(f"  Overall cancellation rate: {(daily_stats['cancelled_flights'].sum() / daily_stats['total_flights'].sum()):.4f}")

# ==============================================================================
# STEP 7: SAVE AGGREGATED DATA
# ==============================================================================
print("\n" + "=" * 80)
print("Step 7: Saving Aggregated Data")
print("=" * 80)

# Save to CSV
output_file = 'flight_cancellations_daily_2024.csv'
daily_stats.to_csv(output_file, index=False)

print(f"\n✓ Saved aggregated data to: {output_file}")
print(f"  Rows: {len(daily_stats)}")
print(f"  Columns: {list(daily_stats.columns)}")

# Get file size
import os
file_size_kb = os.path.getsize(output_file) / 1024
print(f"  File size: {file_size_kb:.1f} KB")

# Display sample output
print(f"\nSample Output (first 10 rows):")
print(daily_stats.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("AGGREGATION COMPLETE")
print("=" * 80)
print("\nOutput file is ready for upload to Google Colab!")
print(f"Size reduction: ~1.2GB → {file_size_kb:.1f}KB ({(1-(file_size_kb/1024)/1200)*100:.2f}% reduction)")
print("\nThis aggregated file enables:")
print("  • Fast upload to Google Colab (free tier)")
print("  • Efficient temporal correlation analysis")
print("  • Daily-level crisis event alignment")
print("  • Reproducible research workflow")
