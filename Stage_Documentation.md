# Three-Stage Intelligence Pipeline Documentation
## Zero-Shot Classification of Crisis-Related News for Forecasting Flight Cancellations

**Team:** Rishiv Bawa, Gaurav Sharma, Veeksith Sumanth  
**Date:** February 6, 2026  
**Research Question:** Can Zero-Shot Classification applied to crisis-related news headlines predict international flight cancellations?

---

## Overview: Noise → Filtering → Intelligence

This document traces the transformation of raw, unstructured data into actionable intelligence through three distinct stages:

```
Stage 1: RAW DATA (Noise)
    ↓
Stage 2: FILTERED & CLASSIFIED (Structured Signals)
    ↓
Stage 3: ACTIONABLE INTELLIGENCE (Insights)
```

---

## STAGE 1: RAW DATA (The Noise)

### What Is This Stage?

Raw data represents the unprocessed, high-volume information streams that contain signal buried within noise. In our pipeline, this consists of:

1. **GDELT Events Database** (Pre-filtered for Google Colab)
   - **Raw Volume:** ~30 million global events per year
   - **Filtered File:** `gdelt_crisis_aviation_clean.csv` (~5MB)
   - **Filtered Records:** 10,122 aviation-related crisis events (loaded), 8,541 after preprocessing
   - **Format:** Structured event records (58 columns)
   - **Content:** Aviation-related crisis events only (pre-filtered from raw GDELT)
   - **Noise Reduction:** 99.96% noise already removed (30M → 10K events)
   - **Optimization:** Pre-filtering eliminates need for multi-GB file processing in Colab

2. **Flight Cancellation Records** (Pre-aggregated for Google Colab)
   - **File:** `flight_cancellations_daily_2024.csv` (~20KB)
   - **Volume:** 366 daily records (Jan 1 - Dec 31, 2024, full year - leap year)
   - **Original Data:** Aggregated from 7,079,081 raw BTS flight records
   - **Format:** Daily operational data (date, total_flights, cancelled_flights, cancellation_rate, is_spike)
   - **Content:** Pre-aggregated U.S. flight operations data from Bureau of Transportation Statistics
   - **Noise Level:** Minimal (all records valid, pre-aggregated at daily level)
   - **Optimization:** Using 20KB aggregated file instead of 1.2GB raw flight data

### Why Is This "Noise"?

- **Lack of Focus:** Raw GDELT includes all global events—only 0.04% relate to aviation crises
- **No Labels:** Events are not categorized by disruption type
- **No Relationships:** No explicit connection between news events and flight operations
- **Temporal Misalignment:** Events are timestamped to the second, flights aggregated daily
- **Scale Problem:** Raw data too large for Google Colab (multi-GB files)

### Google Colab Optimization Strategy

To enable efficient execution on free Google Colab (T4 GPU, 15GB RAM), we use **pre-filtered and pre-aggregated datasets**:

- **GDELT:** Pre-filtered from 30M events → 10,122 aviation crisis events (`gdelt_crisis_aviation_clean.csv`, 5MB)
- **Flight Data:** Pre-aggregated from 7M+ records → 366 daily statistics (`flight_cancellations_daily_2024.csv`, 20KB)
- **Total Upload:** ~5.02MB (vs. 13+ GB raw data)
- **Benefit:** Fast upload, no memory-intensive processing, analysis-ready datasets

### Characteristics of Stage 1 Data

| Property | GDELT Crisis Events | Flight Cancellations (BTS) |
|----------|---------------------|----------------------------|
| **Raw Volume** | 30M rows (raw GDELT) | 7M+ rows (raw BTS) |
| **Filtered/Aggregated File** | `gdelt_crisis_aviation_clean.csv` | `flight_cancellations_daily_2024.csv` |
| **File Size** | ~5 MB | ~20 KB |
| **Records** | 10,122 events (8,541 after preprocessing) | 366 days (Full Year - Leap Year) |
| **Structure** | Semi-structured (58 cols) | Structured (5 cols) |
| **Labeling** | None (pre-filtered only) | None (raw counts) |
| **Temporal Granularity** | Minute-level | Daily |
| **Relevance** | 100% (pre-filtered) | 100% |
| **Pre-processing** | Pre-filtered for Colab | Pre-aggregated for Colab |
| **Data Source** | GDELT Project | BTS On-Time Performance Database |

---

## STAGE 2: FILTERED & CLASSIFIED (Structured Signals)

### What Is This Stage?

Stage 2 transforms raw data into structured, labeled, and temporally aligned signals suitable for analysis. This involves:

1. **Filtering** (Noise Reduction)
2. **Classification** (Signal Labeling)
3. **Clustering** (Pattern Discovery)
4. **Temporal Alignment** (Synchronization)

### 2.1 Filtering: From 30M Events → 11K Relevant Events

**Note:** This filtering was performed as **pre-processing** to create the `gdelt_crisis_aviation_clean.csv` file. The Colab notebook loads this pre-filtered dataset directly, eliminating the need for large-scale data processing during execution.

**Filter 1: Crisis Event Selection (CAMEO Codes)**
- **Criteria:** EventRootCode ∈ {14, 17, 18, 19, 20}
  - 14 = Protest
  - 17 = Coerce
  - 18 = Assault
  - 19 = Fight
  - 20 = Mass Violence
- **Result:** 30M → ~500K events (98.3% noise removed)
- **Justification:** These codes represent disruption-prone scenarios

**Filter 2: Aviation Relevance**
- **Criteria:** Keyword matching in Actor1Name, Actor2Name, ActionGeo_Fullname
- **Keywords:** airport, airline, flight, aviation, runway, terminal, plane, jet, air traffic, ATC, Boeing, Airbus
- **Result:** 500K → 11,113 events (additional 97.8% noise removed)
- **Justification:** Ensures domain specificity to aviation sector

**Filter 3: Text Validity**
- **Criteria:** Event text length ≥ 10 characters
- **Result:** Minimal removal (~0.1%)
- **Justification:** Removes malformed or empty records

**Filter 4: Deduplication**
- **Criteria:** Unique (Day, EventRootCode, Actor1, Actor2, Location)
- **Result:** Removes duplicate mentions of same event
- **Justification:** Avoids counting same event multiple times

**Overall Noise Reduction:** 30M → 10,122 = **99.96% noise removed**

**Output:** `gdelt_crisis_aviation_clean.csv` (5MB, 10,122 records) - ready for Google Colab upload

### 2.2 Classification: Labeling Disruption Types

**Model:** Mistral-Nemo-Instruct-2407 (12B parameters, Q5_K_M quantization, ~7.5GB)  
**Task:** Zero-shot instruction-based classification with grammar-constrained generation  
**Input:** Constructed event text (e.g., "Crisis event: Delta Airlines and security forces involved in incident at JFK Airport. Event type code 18.")  
**Output:** Disruption category + confidence score + reasoning (enforced JSON structure via GBNF)

**Model Selection Rationale:**
- **Reproducibility:** Explicit specification of model weights, seed, temperature, and GBNF grammar constraint
- **Grammar-constrained output (GBNF):** Formal grammar specification guarantees valid JSON structure, eliminating parsing errors
- **Instruction following:** Fine-tuned on instruction datasets for reliable zero-shot classification
- **Quantization efficiency:** Q5_K_M quantization (5-bit) enables GPU inference on Google Colab T4 (free tier) with minimal performance loss
- **Superior reasoning capabilities:** 12B parameters provide nuanced crisis event understanding with detailed explanations
- **Verified deployment:** Implementation validated through peer testing on January 6 Capitol Riots sentiment analysis
- **Processing speed:** ~2-4 seconds per event on Colab T4 GPU with CUDA acceleration

**GBNF Grammar Implementation:**
```python
# JSON grammar (GBNF) enforces structured output
json_grammar_str = r"""
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
array  ::= "[" ws (value ("," ws value)*)? "]" ws
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws ::= ([ \t\n] ws)?
"""
```

**Classification Schema:**
```json
{
  "categories": [
    "extreme_weather_aviation_impact",
    "labor_strike_personnel_shortage", 
    "security_threat_airport_incident",
    "geopolitical_airspace_restriction",
    "infrastructure_technical_failure",
    "natural_disaster_operational_halt",
    "regulatory_grounding_sanction",
    "non_crisis_routine_incident"
  ],
  "confidence_threshold": 0.40
}
```

**Sampling Strategy:**
- Up to 1,000 events per month (balanced temporal representation)
- Total classified: ~8,040 events
- Random seed: 42 (reproducibility)
- Inference time: ~2-4 seconds per event on GPU (Colab T4)
- Total processing time: ~40-60 minutes for 8,040 events
- Checkpointing: Auto-save every 500 events for crash recovery

**Classification Output:**
- Each event labeled with primary disruption type (from 8 categories)
- Confidence score (0.0-1.0) self-assessed by LLM with grammar-constrained JSON output (GBNF ensures validity)
- Reasoning field explaining the classification decision
- Events below 0.40 threshold marked as "low_confidence"
- Error rate: <0.1% (GBNF grammar prevents malformed outputs)

**Example Results:**

| Event Text | Disruption Type | Confidence |
|------------|-----------------|------------|
| "Crisis event: Airline workers and management in protest at Heathrow..." | labor_strike_personnel_shortage | 0.78 |
| "Crisis event: Hurricane and air traffic control at Miami Airport..." | extreme_weather_aviation_impact | 0.85 |
| "Crisis event: Security personnel and suspicious individual at CDG..." | security_threat_airport_incident | 0.72 |

### 2.3 Clustering: Discovering Latent Themes

**Purpose:** Identify thematic patterns not captured by predefined labels

**Method:** Community detection via graph-based clustering

**Implementation:**
1. **Embedding:** Convert event_text to 384-dimensional vectors using `all-MiniLM-L6-v2`
2. **Similarity:** Compute pairwise cosine similarity (N×N matrix)
3. **Graph Construction:** Create edges where similarity > 0.5
4. **Community Detection:** Apply Louvain algorithm (resolution=1.0)
5. **Labeling:** Manual inspection of representative texts per cluster

**Output:**
- 5-8 thematic clusters with interpretable labels (Top 3 largest used for correlation analysis):
  - **Cluster -1:** Uncategorized/Noise (events with low similarity to any cluster)
  - **Example identified clusters:**
    - **AIRLINE Incidents** (general airline operations and service disruptions)
    - **Military Aviation Conflicts** (fighter jets, military airspace operations)
    - **Airport Security Operations** (police, security personnel, safety incidents)
  - Additional smaller clusters identified via pattern-based labeling
- Cluster assignments for each event
- Intra-cluster similarity scores
- Dominant disruption type within each cluster
- Automated and manual labeling based on actor patterns, locations, and event codes

**Computational Efficiency Note:** To balance feature richness with practical analysis, only the **Top 3 largest clusters** (by event count, excluding noise) are included in temporal aggregation and correlation analysis. This ensures statistical significance while maintaining computational feasibility.

**Value Added:**
- Reveals emerging crisis types not in predefined taxonomy
- Validates classification schema (do clusters align with labels?)
- Enables unsupervised pattern discovery

### 2.4 Temporal Alignment: Synchronizing Data Sources

**Challenge:** GDELT events are timestamped at minute-level, flights at day-level

**Solution:** Aggregate crisis events to daily counts by disruption type

**Process:**
1. **Convert GDELT timestamps to date (YYYY-MM-DD):** Normalize minute-level timestamps to daily granularity
2. **Group events by (date, disruption_type):** Create multi-index grouping for temporal and categorical dimensions
3. **Count events per group:** Generate daily count matrix with disruption types as columns
4. **Merge with daily flight cancellation rates on date key:** Left join to preserve all flight operation days

**Implementation Details:**
- Valid classifications filtered (confidence ≥ 0.40, excluding 'low_confidence' and 'error' categories)
- Wide-format pivot table created with disruption types as separate columns
- Total crisis events per day calculated as sum across all disruption types
- Missing values (days with no crisis events) filled with zeros
- **Cluster features added:** Thematic cluster densities from community detection included as additional predictive features
- Final dataset: 366 rows (Full Year 2024) × (8 disruption types + cluster features + flight metrics)

**Result:** Unified dataset with structure (sample columns shown with abbreviated headers):

| Date | Total Flights | Cancelled | Cancel Rate | Weather | Strike | Security | Infrastructure | Cluster_0 | Cluster_1 | Total Crisis |
|------|--------------|-----------|-------------|---------|--------|----------|----------------|-----------|-----------|--------------|
| 2024-01-01 | 2,384 | 85 | 0.0357 | 0 | 0 | 22 | 1 | 15 | 8 | 23 |
| 2024-01-02 | 2,131 | 36 | 0.0169 | 0 | 0 | 30 | 0 | 25 | 5 | 30 |

*Note: Full column names use descriptive labels (e.g., `extreme_weather_aviation_impact`, `labor_strike_personnel_shortage`, etc.). Table shows abbreviated headers for readability.*

**Temporal Coverage:**
- Total days: 366 (full year 2024 - leap year)
- Days with crisis signals: ~275-300 (75-82%)
- Days without crisis signals: ~66-91 (18-25%)
- Mean daily crisis events: Variable by disruption type and cluster theme

### Stage 2 Output Summary

| Metric | Value |
|--------|-------|
| **Input (Stage 1)** | 30M crisis events + 366 days real BTS flight records (Full Year 2024) |
| **Output (Stage 2)** | 8,040 classified events + 366 daily aggregations with dual features |
| **Noise Removed** | 99.97% (30M → 8,541 preprocessed → 8,040 classified) |
| **Labels Added** | 8 disruption categories (Mistral-Nemo classification) |
| **Clusters Identified** | 5-8 thematic groups (Top 3 used for correlation) |
| **Temporal Resolution** | Daily (366 data points for Full Year 2024 - leap year) |
| **Feature Sets** | 8 disruption types + Top 3 cluster themes |
| **Files Generated** | `crisis_events_classified.csv`, `crisis_events_clustered.csv`, `intelligence_dataset.csv` |

---

## STAGE 3: ACTIONABLE INTELLIGENCE (The Insights)

### What Is This Stage?

Stage 3 converts structured signals into **actionable insights** that answer the research question and enable decision-making.

### Research Question (Restated)
**Can Zero-Shot Classification applied to crisis-related news headlines predict international flight cancellations?**

### Hypothesis (Restated)
**Crisis event signals from zero-shot classification exhibit statistically significant temporal correlation (r ≥ 0.15, p < 0.05) with flight cancellation rates at lag windows of 0-3 days, enabling early warning identification.**

### 3.1 Analysis Method: Lagged Correlation

**Purpose:** Determine if crisis event signals predict future cancellations

**Method:**
- Compute Pearson correlation between crisis event counts (time t) and cancellation rates (time t+lag)
- Test lags: 0, 1, 2, 3 days
- Calculate p-values to test statistical significance (H₀: r=0)
- Analyze BOTH disruption types (8 categories) AND thematic clusters (Top 3)

**Why Lagged?**
- Crisis events may have delayed impact (e.g., strike announced Monday → cancellations Tuesday)
- Lead time is critical for actionable forecasting

**Dual-Feature Approach:**
- **Disruption Types:** Predefined categories from zero-shot classification
- **Thematic Clusters:** Emergent patterns from community detection
- **Total features analyzed:** 11+ features (8 disruption types + 3 cluster themes + aggregate)

**Key Findings from Full-Year Analysis (366 days):**

| Feature Type | Best Feature | Optimal Lag | Correlation | P-value |
|--------------|--------------|-------------|-------------|---------|
| Disruption Type | geopolitical_airspace_restriction | 2 days | r=+0.216 | p<0.0001*** |
| Cluster Theme | AIRLINE Incidents | 0 days | r=+0.190 | p<0.01** |
| Aggregate | total_crisis_events | 1 day | r=+0.150 | p<0.05* |

*Statistical significance: * p<0.05  ** p<0.01  *** p<0.001

**Hypothesis Evaluation:** ✓ **SUPPORTED**  
Multiple features exceed r=0.15 with statistical significance (p<0.05), demonstrating that crisis signals provide meaningful predictive value for flight cancellation patterns.

### 3.2 Complementary Metrics: Precision, Recall, and ROC-AUC

### 3.2 Complementary Metrics: Precision, Recall, and ROC-AUC

**Purpose:** While correlation demonstrates predictive relationships, classification metrics (precision/recall/ROC-AUC) provide operational insights for real-world deployment.

**Method:**
- **Target:** Days with cancellation rates >90th percentile (9 spike days / 366 total)
- **Predictors:** Crisis event counts for each feature (disruption types + clusters)
- **Threshold Optimization:** For each feature, find optimal crisis count threshold that maximizes F1-score
- **Metrics Computed:**
  - **Precision:** When crisis signals are elevated, how often do spikes occur?
  - **Recall:** Of all spike days, how many were preceded by crisis signals?
  - **F1-Score:** Harmonic mean balancing precision and recall
  - **ROC-AUC:** Discriminative power (ability to distinguish spike vs. non-spike days)

**Key Results:**

| Metric | Best Feature | Value | Interpretation |
|--------|--------------|-------|----------------|
| Best F1-Score | geopolitical_airspace_restriction (lag 2) | 0.42 | Moderate balance of precision/recall |
| Best Precision | infrastructure_technical_failure (lag 1) | 0.35-0.45 | ~35-45% of crisis signals correctly predict spikes |
| Best Recall | total_crisis_events (lag 0) | 0.65-0.75 | ~65-75% of spike days had crisis signals |
| Best ROC-AUC | geopol itical_airspace_restriction (lag 2) | 0.68-0.72 | Weak-to-acceptable discrimination |

**Class Imbalance Note:**
- Spike days: 9/366 (2.5%) - severe class imbalance
- This imbalance inherently limits precision even when correlation is significant
- ROC-AUC and correlation are more robust metrics for imbalanced datasets

**Interpretation:**
- **Precision <70%** indicates high false alarm rate (expected given 97.5% non-spike days)
- **Correlation r>0.15** (statistically significant) demonstrates genuine predictive relationship
- **ROC-AUC 0.68-0.72** shows signals provide meaningful discrimination above random chance (0.5)
- Crisis signals are **useful for early warning** but require additional context for operational deployment

### 3.3 Actionable Insights

**Insight 1: Geopolitical Airspace Restrictions Provide Strongest Forecast**
- **Finding:** Geopolitical crisis events correlate with cancellations 2 days later (r=+0.216, p<0.0001)
- **Action:** Monitor international tensions, military conflicts, airspace closures 48 hours in advance
- **Stakeholder:** Airline route planning, international operations teams, travel advisories
- **Operational Value:** 2-day lead time enables proactive route adjustments and customer notifications

**Insight 2: AIRLINE Incidents Cluster Shows Same-Day Impact**
- **Finding:** General airline incident cluster correlates with cancellations at lag=0 (r=+0.190)
- **Action:** Real-time monitoring of airline operational disruptions for same-day response
- **Stakeholder:** Airline operations centers, customer service teams
- **Operational Value:** Enables rapid response and passenger accommodation

**Insight 3: Infrastructure Failures Show Moderate Predictive Power**
- **Finding:** Infrastructure technical failures correlate across 0-2 day lags
- **Action:** Extended monitoring for cascading effects of system outages, equipment failures
- **Stakeholder:** Airport facility management, technical operations
- **Operational Value:** Allows preparation for extended disruption periods

**Insight 4: Dual-Feature Approach Captures Both Expected and Emergent Patterns**
- **Finding:** Classification features (disruption types) AND clustering features (thematic patterns) both contribute predictive value
- **Action:** Deploy hybrid monitoring system combining predefined categories with data-driven pattern detection
- **Stakeholder:** Intelligence analysts, predictive modeling teams
- **Operational Value:** Comprehensive early warning system that adapts to emerging crisis types

### 3.4 Visualization: Correlation Heatmap

The generated `correlation_heatmap.png` visualizes:
- **X-axis:** Lag (0, 1, 2, 3 days)
- **Y-axis:** Disruption type
- **Color:** Correlation strength (red=negative, green=positive)
- **Annotations:** Exact r-values

This single visualization communicates the entire predictive landscape to stakeholders.

### 3.5 What Makes This "Intelligence"?

Stage 3 outputs are intelligence because they:

1. **Answer a Specific Question:** "Can we predict cancellations?" → Yes, with labor strikes at 2-day lag
2. **Enable Action:** Operations teams can preemptively adjust staffing based on crisis signals
3. **Quantify Uncertainty:** Correlation coefficients and p-values provide confidence levels
4. **Reveal Non-Obvious Patterns:** Lag effects not apparent in raw data
5. **Support Decision-Making:** Airlines can choose to:
   - Increase customer service staff before predicted spikes
   - Waive rebooking fees proactively
   - Adjust pricing models based on risk

### Stage 3 Output Summary

| Output | Description | Stakeholder Use |
|--------|-------------|-----------------|
| **Correlation Table** | r-values and p-values by disruption type and lag | Data scientists, researchers |
| **Precision Score** | 40% (hypothesis test result) | Executive summary |
| **Heatmap Visualization** | Color-coded correlation matrix | Executive briefings |
| **Recommendations** | "Monitor labor strikes 2 days ahead" | Operations teams |
| **Lead Time Estimate** | 24-48 hours for labor disruptions | Strategic planning |

---

## Pipeline Summary: Complete Transformation

### Stage 1 → Stage 2 Transition
- **Input Files:** `gdelt_crisis_aviation_clean.csv` (5MB, pre-filtered) + `flight_cancellations_daily_2024.csv` (20KB, pre-aggregated)
- **Process:** Phi-3.5 Mini classification + thematic clustering + temporal alignment
- **Output:** 11K classified events + 366 daily aggregations with dual features (8 disruption types + cluster themes)
- **Transformation:** **Pre-filtered Data → Classified Signals** (ready for correlation analysis)

### Stage 2 → Stage 3 Transition
- **Input:** 11K classified events with disruption labels and cluster assignments
- **Process:** Temporal aggregation, dual-feature correlation analysis (types + clusters), precision testing
- **Output:** Correlation coefficients for both feature sets, lead time estimates, actionable recommendations
- **Transformation:** **Structured Signals → Actionable Intelligence**

### End-to-End Pipeline Metrics

| Metric | Value |
|--------|-------|
| **Input Files** | `gdelt_crisis_aviation_clean.csv` (5MB) + `flight_cancellations_daily_2024.csv` (20KB) |
| **Total Upload Size** | ~5.02 MB (optimized for Google Colab) |
| **Input Events** | 10,122 pre-filtered aviation crisis events + 366 days flight data |
| **Preprocessed Events** | 8,541 events (after deduplication and text validation) |
| **Classified Events** | 8,040 events (sampled for balanced temporal representation) |
| **Final Intelligence Volume** | 366 daily insights + dual-feature correlation matrices |
| **Processing Time (Colab T4 GPU)** | ~40-60 minutes (classification with checkpointing) |
| **Reproducibility** | Fully reproducible (seed=42, temperature=0.0, GBNF grammar) |
| **Hypothesis Validation** | ✓ SUPPORTED (correlation-based: r≥0.15, p<0.05) |
| **Model** | Mistral-Nemo-Instruct-2407 (12B parameters, Q5_K_M quantization, ~7.5GB) |
| **Pre-processing Approach** | Data pre-filtered offline; Colab executes classification & analysis only |
| **Feature Engineering** | Dual-feature approach: 8 disruption types + 3 thematic clusters |
| **Evaluation Metrics** | Correlation (primary), Precision/Recall (complementary), ROC-AUC (discrimination) |

---

## Limitations & Future Work

### Current Limitations

1. **Full-Year Temporal Coverage:** 366-day analysis period (Jan-Dec 2024 - full year); enables comprehensive seasonal pattern detection and robust statistical analysis across all disruption types
2. **Classified Sample Size:** 8,040 events (sampled for balanced temporal representation); computational constraints limited exhaustive classification of all 10,122 loaded events
3. **Class Imbalance:** Severe imbalance (9 spike days / 366 total = 2.5%) inherently limits precision metrics; correlation and ROC-AUC provide more robust evaluation
4. **Geographic Scope:** U.S. domestic flights only (BTS data limitation); international routes may exhibit different crisis-cancellation patterns
5. **Temporal Granularity:** Daily aggregation smooths intra-day patterns; hourly analysis could reveal finer-grained relationships
6. **Model Size Trade-off:** Mistral-Nemo-12B balances performance and efficiency; larger models (70B+) may improve classification accuracy but exceed free Colab GPU limits

### Future Work Directions

1. **Extended Temporal Coverage:** Multi-year analysis (2020-2024) to capture pandemic recovery patterns and long-term trends
2. **International Expansion:** Incorporate global flight data and GDELT events from all regions for worldwide applicability
3. **Real-Time Pipeline:** Deploy streaming classification system for operational early warning (API integration with GDELT and BTS live feeds)
4. **Advanced Models:** Experiment with larger instruction-tuned models (e.g., Mixtral-8x22B) and compare against fine-tuned classifiers
5. **Causal Inference:** Move beyond correlation to establish causal relationships using methods like Granger causality or intervention analysis
6. **Multi-Modal Integration:** Combine news signals with weather data, social media sentiment, and ADS-B flight tracking for comprehensive disruption forecasting
7. **Operational Deployment:** Partner with airlines/airports to validate predictions in real-world operations and measure business impact
3. **Lead Time Ambiguity:** Cannot distinguish "forecasted" vs "active" crises (temporal phase classification needed)
4. **Geographic Specificity:** No country/region-level analysis; global patterns aggregated
5. **Seasonal Bias:** Jan-Feb data may not represent annual patterns (winter-specific disruptions)

### Future Enhancements

1. **Multi-Year Analysis:** Expand to multi-year analysis (2020-2025) for long-term trends and year-over-year pattern comparison
2. **Temporal Phase Classification:** Add "warning/active/aftermath" labels to distinguish event stages and improve lead time estimation
3. **Geographic Clustering:** Identify region-specific disruption patterns (country/continent-level analysis)
4. **RAG Pipeline:** Implement real-time crisis monitoring with vector database retrieval for live event streaming
5. **Fine-grained Temporal Resolution:** Test hour-level or flight-level aggregation for more precise correlation analysis
6. **Multi-modal Features:** Incorporate social media sentiment, weather data, and economic indicators as additional predictive features

---

## Conclusion

This pipeline demonstrates the feasibility of using Phi-3.5 Mini (3.8B parameters) for zero-shot classification on open-source crisis news to generate predictive signals for flight cancellations. Analysis was conducted using **real BTS flight data** (Full Year 2024, 366 days) integrated with GDELT crisis events. The methodology is sound and replicable with full-year temporal coverage enabling robust statistical analysis. Key findings include:

- **Labor strikes provide 2-day advance warning** (r=+0.41, p<0.001)
- **Weather and security events show 0-1 day correlations**
- **Infrastructure failures have persistent multi-day effects**
- **Dual-feature approach** (disruption types + cluster themes) provides complementary predictive signals

With extended temporal coverage and refined phase classification, this approach has potential for operational deployment in aviation risk management systems.

---

**Files Required for Google Colab Execution:**
1. `Crisis_Flight_Cancellation_Pipeline.ipynb` - Main analysis notebook
2. `gdelt_crisis_aviation_clean.csv` - Pre-filtered crisis events (5MB input)
3. `flight_cancellations_daily_2024.csv` - Pre-aggregated flight data (20KB input)

**Files Generated During Execution:**
- `crisis_events_preprocessed.csv` - Preprocessed events with constructed text
- `crisis_events_classified.csv` - Stage 2 output with Phi-3.5 Mini classifications
- `crisis_events_clustered.csv` - Stage 2 output with cluster assignments
- `intelligence_dataset.csv` - Stage 2-3 merged with dual features
- `correlation_analysis.csv` - Stage 3 statistics
- `correlation_heatmap.png` (Stage 3 visualization)

**Reproducibility:** All code available in `Crisis_Flight_Cancellation_Pipeline.ipynb` with fixed random seed (42), greedy decoding (temperature=0.0), and microsoft/Phi-3.5-mini-instruct model from HuggingFace Hub.
