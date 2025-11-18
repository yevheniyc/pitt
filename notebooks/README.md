# Source Notebooks

This directory contains the essential Jupyter notebooks that served as source material for the professional publications in `../publications/`.

## 📁 Organization

**Essential Notebooks (6 files, 37.5 MB):**

- **project_1_eda/** - Exploratory Data Analysis

  - `exploratory_data_analysis.ipynb` (10M) - Comprehensive EDA with 28+ visualizations

- **project_2_machines/** - Manufacturing Analytics

  - `manufacturing_analysis.ipynb` (9.6M) - Multi-source data integration, 206+ visualizations, quality testing

- **project_3_spotify/** - Spotify Popularity Prediction

  - `spotify_main_analysis.ipynb` (5.1M) - Complete ML analysis with findings summary
  - `spotify_clustering.ipynb` (11M) - Detailed K-means clustering analysis

- **project_1_network_analysis/** - Network Science

  - `network_centrality_analysis.ipynb` (396K) - Graph theory and centrality metrics

- **project_2_map_visualization/** - Geospatial Analytics

  - `geospatial_visualization.ipynb` (1.4M) - Interactive mapping with Folium

- **project_3_data_story_final/** - Fire Safety Dashboard
  - `fire_safety_dashboard.py` (1.6M) - Gradio app with Plotly/Folium visualizations
  - `updated_analysis.py` - Data quality correction script
  - `data/` - 930K+ dispatch records (2015-2024)

## 🎯 Purpose

These notebooks are **working source files** containing:

- Complete exploratory analysis
- All experimental visualizations
- Iterative code development
- Detailed research notes

**Note**: Redundant tutorial notebooks and small research files have been removed for clarity.

## 📄 Publications

**Polished, publication-ready versions** are available as professional PDFs in `../publications/`:

- `spotify_popularity/` → Based on project_1_eda + project_3_spotify
- `manufacturing_analytics/` → Based on project_2_machines
- `network_analysis/` → Based on project_1_network_analysis
- `fire_safety_dashboard/` → Based on project_3_data_story_final

## 🔧 Usage

View notebooks:

```bash
uv run jupyter lab
```

Generate publications:

```bash
cd ../
make pdf [publication_name]
```

---

**Purpose**: Source material for academic research publications
**Status**: Comprehensive analysis notebooks supporting 4 professional publications
