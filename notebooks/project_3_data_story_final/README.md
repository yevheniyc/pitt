# Fire Safety Data Story - Interactive Dashboard

## Overview

Interactive data analysis dashboard examining fire safety patterns in Allegheny County using 930,000+ emergency dispatch records (2015-2024).

## Files Structure

```
project_3_data_story_final/
├── data/                                    # Data files
│   ├── fire_dispatches_fresh.csv          # Original WPRDC dataset
│   ├── allegheny_fire_dispatches.csv      # Processed dataset
│   └── corrected_fire_alarms.csv          # Generated corrected data
├── fire_safety_dashboard.py               # Main interactive dashboard
├── updated_analysis.py                    # Data correction script
└── README.md                              # This file
```

## Quick Start

### Prerequisites

```bash
pip install pandas plotly folium geopandas gradio numpy
```

### Run the Application

```bash
# Generate corrected data (first time only)
python updated_analysis.py

# Launch interactive dashboard
python fire_safety_dashboard.py
```

The dashboard will launch in your browser with interactive visualizations.

## Key Features

### Interactive Visualizations

- **Geographic Analysis**: Incident heatmaps and municipal distributions
- **Temporal Trends**: Year-over-year and seasonal patterns
- **Emergency Priorities**: Response urgency analysis
- **False Alarm Analysis**: Cost impact and prevention insights

### Data Insights

- **37.3%** of incidents are fire alarms (false alarm crisis)
- **Geographic disparities** in incident distribution
- **Seasonal patterns** vary by incident type
- **~$1,000 cost** per false alarm response

### Policy Recommendations

1. **Smart Alarm Technology**: 30-50% false alarm reduction potential
2. **Community Prevention**: Target high-risk neighborhoods
3. **Seasonal Preparedness**: Deploy resources based on patterns

## Data Source

- **Provider**: Western Pennsylvania Regional Data Center (WPRDC)
- **Dataset**: Allegheny County 911 Fire Dispatches
- **Timeline**: 2015-2024 (10 years)
- **Records**: 930,808 total incidents

## Technical Stack

- **Frontend**: Gradio (Python web framework)
- **Visualization**: Plotly (interactive charts), Folium (maps)
- **Data Processing**: Pandas, NumPy
- **Geospatial**: GeoPandas for geographic analysis
