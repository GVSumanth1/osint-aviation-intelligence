# =========================
# RESUME CLASSIFICATION: Events 2500 to 8040
# =========================
# This cell continues classification from event 2500 through 8040
# IMPORTANT: Requires RANDOM_SEED=42 to maintain event order consistency

import os
import time
import pandas as pd

print("="*80)
print("RESUMING CLASSIFICATION FROM EVENT 2500")
print("="*80 + "\n")

# Load checkpoint file with first 2500 classified events
checkpoint_file = os.path.join(DRIVE_PATH, "checkpoint_2500.csv")
try:
    df_completed = pd.read_csv(checkpoint_file)
    print(f"Loaded checkpoint: {len(df_completed)} events from {checkpoint_file}")
except FileNotFoundError:
    print("Checkpoint file not found. Attempting to load from classified file...")
    alt_file = os.path.join(DRIVE_PATH, "crisis_events_classified.csv")
    df_completed = pd.read_csv(alt_file)
    df_completed = df_completed.iloc[:2500]
    print(f"Loaded first 2500 events from: {alt_file}")

# Load original unclassified df_sample from checkpoint
# This preserves the exact event order from the original sampling
checkpoint_sample_file = os.path.join(DRIVE_PATH, "df_sample_original.csv")

if os.path.exists(checkpoint_sample_file):
    # Load saved unclassified sample
    df_original_sample = pd.read_csv(checkpoint_sample_file)
    df_original_sample['date'] = pd.to_datetime(df_original_sample['date'])
    print(f"Loaded original df_sample: {len(df_original_sample)} events")
else:
    # Reconstruct df_sample using same random seed (guarantees identical order)
    print("Reconstructing df_sample with RANDOM_SEED=42 (identical to original run)...")
    
    CRISIS_FILE = DRIVE_PATH + 'gdelt_crisis_aviation_clean.csv'
    df_crisis = pd.read_csv(CRISIS_FILE)
    df_crisis['Day'] = df_crisis['Day'].astype(str)
    df_crisis = df_crisis[df_crisis['Day'].str.startswith('2024')].copy()
    
    # Preprocessing - EXACT same logic as Pipeline_5
    df_crisis['date'] = pd.to_datetime(df_crisis['Day'], format='%Y%m%d', errors='coerce')
    df_crisis = df_crisis[df_crisis['date'].notna()].copy()
    
    # Build event_text using same function as Pipeline_5
    def build_event_text(row):
        actor1 = str(row.get('Actor1Name', '')).strip()
        actor2 = str(row.get('Actor2Name', '')).strip()
        location = str(row.get('ActionGeo_Fullname', '')).strip()
        code = str(row.get('EventRootCode', '')).strip()
        text = f"Crisis event: {actor1} and {actor2} involved in incident at {location}. Event type code {code}."
        return text
    
    df_crisis['event_text'] = df_crisis.apply(build_event_text, axis=1)
    
    # Text validity filter
    df_crisis = df_crisis[df_crisis['event_text'].str.len() >= 10].copy()
    
    # Deduplication
    df_crisis = df_crisis.drop_duplicates(
        subset=['Day', 'EventRootCode', 'Actor1Name', 'Actor2Name', 'ActionGeo_Fullname'],
        keep='first'
    )
    
    # Sampling with fixed random seed
    df_crisis['month'] = df_crisis['date'].dt.to_period('M')
    sampled_events = []
    for month, group in df_crisis.groupby('month'):
        sample_size = min(1000, len(group))
        sampled = group.sample(n=sample_size, random_state=RANDOM_SEED)
        sampled_events.append(sampled)
    
    df_original_sample = pd.concat(sampled_events, ignore_index=True)
    print(f"Reconstructed df_sample: {len(df_original_sample)} events (order guaranteed by seed=42)")

# Extract remaining events to classify
df_remaining = df_original_sample.iloc[2500:8040].copy()
print(f"\nRemaining events to classify: {len(df_remaining)}")
print(f"Date range: {df_remaining['date'].min()} to {df_remaining['date'].max()}")

# Classification loop
if llm is not None:
    print("\n" + "="*80)
    print(f"STARTING CLASSIFICATION: {len(df_remaining)} EVENTS")
    print("="*80)
    print(f"Model: Mistral-Nemo-Instruct-2407 (12B)")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Progress updates: every 50 events")
    print(f"Checkpoints: every 500 events\n")
    
    classifications = []
    start_time = time.time()
    completed_count = 0
    
    for idx, row in df_remaining.iterrows():
        event_text = str(row['event_text'])
        
        # Classify event
        label, confidence, reasoning = classify_with_mistral(event_text, CONFIDENCE_THRESHOLD)
        
        classifications.append({
            'label': label,
            'confidence': confidence,
            'reasoning': reasoning
        })
        
        completed_count += 1
        
        # Progress tracking every 50 events
        if completed_count % 50 == 0:
            elapsed_time = time.time() - start_time
            rate = completed_count / elapsed_time if elapsed_time > 0 else 0
            eta = (len(df_remaining) - completed_count) / rate / 60 if rate > 0 else 0
            total_done = 2500 + completed_count
            print(f"Processed {total_done}/8040 events... (Latest: {label}, conf: {confidence:.2f})")
        
        # Checkpoint saving
        if completed_count % 500 == 0:
            print(f"\nCHECKPOINT: Saving progress at {2500 + completed_count} events...")
            df_temp = df_remaining.iloc[:completed_count].copy()
            df_temp['disruption_type'] = [c['label'] for c in classifications]
            df_temp['confidence'] = [c['confidence'] for c in classifications]
            df_temp['classification_reasoning'] = [c['reasoning'] for c in classifications]
            
            df_combined = pd.concat([df_completed, df_temp], ignore_index=True)
            checkpoint_file_new = os.path.join(DRIVE_PATH, f"checkpoint_{2500 + completed_count}.csv")
            df_combined.to_csv(checkpoint_file_new, index=False)
            print(f"Saved checkpoint: {checkpoint_file_new}\n")
    
    # Add classification results
    df_remaining['disruption_type'] = [c['label'] for c in classifications]
    df_remaining['confidence'] = [c['confidence'] for c in classifications]
    df_remaining['classification_reasoning'] = [c['reasoning'] for c in classifications]
    
    # Combine all events
    df_final = pd.concat([df_completed, df_remaining], ignore_index=True)
    
    # Save final results
    output_file = os.path.join(DRIVE_PATH, "crisis_events_classified_complete.csv")
    df_final.to_csv(output_file, index=False)
    
    # Display summary
    print("\n" + "="*80)
    print("CLASSIFICATION COMPLETE - ALL 8040 EVENTS PROCESSED")
    print("="*80)
    print(f"\nTotal events: {len(df_final)}")
    print(f"  From checkpoint: 2500")
    print(f"  Newly processed: {len(df_remaining)}")
    
    print("\n" + "-"*80)
    print("Disruption Type Distribution:")
    print("-"*80)
    print(df_final['disruption_type'].value_counts())
    
    print("\n" + "-"*80)
    print("Confidence Score Statistics:")
    print("-"*80)
    print(df_final['confidence'].describe())
    
    print("\n" + "-"*80)
    print("Files Saved:")
    print("-"*80)
    print(f"  {output_file}")
    
    print("\n" + "-"*80)
    print("Sample Results (events 2500-2505):")
    print("-"*80)
    display_cols = ['date', 'event_text', 'disruption_type', 'confidence']
    print(df_final.iloc[2500:2505][display_cols].to_string())
    
    # Update df_sample for subsequent cells
    df_sample = df_final.copy()
    print("\nUpdated df_sample variable for downstream analysis")
    
else:
    print("\nERROR: Mistral model (llm) not loaded")
    print("Please run the model loading cell first")

print("\n" + "="*80)
print("END OF RESUMPTION CELL")
print("="*80)
