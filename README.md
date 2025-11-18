# University of Pittsburgh - AI Master's Research Portfolio

**Professional academic publications from data science coursework**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Quarto](https://img.shields.io/badge/Made%20with-Quarto-blue)](https://quarto.org)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)

## 🎯 Overview

This repository showcases professional academic publications from Master's coursework in Artificial Intelligence at the University of Pittsburgh. Instead of sharing raw Jupyter notebooks, these works are presented as polished research papers using Quarto for publication-quality PDF output.

## 📚 Publications

### 1. Predicting Song Popularity on Spotify
**Machine Learning & Statistical Modeling**

A comprehensive study using logistic regression and interaction modeling to predict song popularity from audio features across six genres.

- **Dataset**: Spotify songs with 14 audio features
- **Methods**: EDA, clustering, progressive logistic regression, cross-validation
- **Key Finding**: Genre + audio feature interactions achieve ROC AUC 0.675
- **Skills**: Python, pandas, scikit-learn, statistical modeling

📄 [View Publication](publications/spotify_popularity/)

---

### 2. Data-Driven Fire Safety Analytics
**Applied Analytics & Public Safety Intelligence**

Analysis of 930,808 emergency dispatch records from Allegheny County to inform fire safety policy and resource allocation.

- **Dataset**: 10 years of 911 fire dispatch data (2015-2024)
- **Methods**: Geospatial analysis, temporal decomposition, interactive dashboard
- **Key Finding**: 37.3% of dispatches are fire alarms; $225M false alarm cost over 10 years
- **Skills**: Python, geopandas, Gradio, public policy analysis

📄 [View Publication](publications/fire_safety_dashboard/)

---

### 3. Manufacturing Process Analytics
**Data Engineering & Industrial IoT**

Multi-source data integration demonstrating PCA and clustering for manufacturing operational intelligence.

- **Dataset**: Sensor data from 3 machines + supplier information (45,000 observations)
- **Methods**: Multi-source merging, PCA, K-means clustering
- **Key Finding**: 5 operational regimes; supplier quality correlates with machine stability
- **Skills**: Python, pandas, scikit-learn, data integration

📄 [View Publication](publications/manufacturing_analytics/)

---

### 4. Network Centrality in College Football
**Network Science & Graph Theory**

Application of network centrality metrics to analyze competitive positioning in college football.

- **Dataset**: 115 Division I teams, 892 games (2022-2023 season)
- **Methods**: Graph construction, degree/betweenness centrality, NetworkX
- **Key Finding**: Penn State (breadth) vs. Ohio State (bridge position) - complementary centrality profiles
- **Skills**: Python, NetworkX, graph theory, sports analytics

📄 [View Publication](publications/network_analysis/)

---

## 🛠️ Technical Stack

**Analysis & Modeling:**
- Python 3.8+
- pandas, numpy
- scikit-learn (ML)
- NetworkX (graphs)
- geopandas, folium (geospatial)

**Visualization:**
- matplotlib, seaborn
- plotly (interactive)
- Gradio (dashboards)

**Publication:**
- Quarto (PDF generation)
- LaTeX (professional formatting)
- BibTeX (citations)

## 📖 Building Publications

### Prerequisites

```bash
# Install Quarto
brew install --cask quarto

# Install Python dependencies
pip install -r requirements.txt
```

### Generate PDFs

```bash
# Generate all publications
make pdf

# Generate specific publication
make pdf spotify_popularity
make pdf fire_safety_dashboard
make pdf manufacturing_analytics
make pdf network_analysis

# View available commands
make help
```

Output PDFs appear in `publications/pdf/`

## 📂 Repository Structure

```
pitt/
├── publications/              # 📄 Academic publications (Quarto .qmd)
│   ├── spotify_popularity/
│   ├── fire_safety_dashboard/
│   ├── manufacturing_analytics/
│   ├── network_analysis/
│   └── pdf/                  # Generated PDFs
├── notebooks/                 # 📓 Source Jupyter notebooks (not public)
├── Makefile                   # Build commands
├── .gitignore                 # Excludes courses/ directory
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🎓 Academic Context

**Institution**: University of Pittsburgh
**Program**: Master of Science in Data Science (MSDS)
**Emphasis**: Applied Artificial Intelligence
**Focus Areas**: Machine Learning, Data Analytics, Network Science, Geospatial Analysis

**Courses:**
- Introduction to Data Science & Computing
- Data Visualization
- Predictive Modeling

## 🏆 Skills Demonstrated

**Data Science:**
- Multi-source data integration
- Exploratory data analysis
- Feature engineering
- Statistical modeling

**Machine Learning:**
- Logistic regression
- K-means clustering
- Principal Component Analysis
- Cross-validation

**Specialized:**
- Geospatial analysis
- Network science
- Time series analysis
- Interactive dashboards

**Engineering:**
- Data pipeline development
- Reproducible research
- Version control
- Professional documentation

## 📊 Research Philosophy

These publications follow three principles:

1. **Show, Don't Tell**: Concrete examples over abstract descriptions
2. **Data-Driven Narrative**: Full stories backed by real metrics and code
3. **Professional Quality**: Publication-grade visualizations and LaTeX formatting

## 📝 Citation

If referencing this work:

```bibtex
@misc{chuba2025pitt,
  author = {Chuba, Yevheniy},
  title = {University of Pittsburgh AI Master's Research Portfolio},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yevheniyc/pitt}
}
```

## 📧 Contact

**Author**: Yevheniy Chuba
**Institution**: University of Pittsburgh
**Program**: Master of Science in Data Science (MSDS)
**LinkedIn**: [yev-chuba](https://www.linkedin.com/in/yev-chuba-57518434/)
**GitHub**: [yevheniyc](https://github.com/yevheniyc)

## 📜 License

This work is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

---

**Last Updated**: November 2025  
**Version**: 1.0.0

_Academic research portfolio showcasing data science and machine learning expertise through professional publications._
