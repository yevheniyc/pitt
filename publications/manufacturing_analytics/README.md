# Manufacturing Process Analytics

**Multi-Source Data Integration for Operational Intelligence**

## 📊 Overview

This project demonstrates data engineering and machine learning techniques for manufacturing analytics. We integrated sensor data from three machines with supplier information to identify operational patterns and supplier performance relationships.

## 🎯 Key Achievements

- **Data Integration**: Merged 5 separate CSV files (45,000 total observations)
- **PCA**: Reduced 4 sensor variables to 2 components retaining 73% variance
- **Clustering**: Identified 5 distinct operational regimes using K-means
- **Supplier Analysis**: Discovered quality correlations with operating stability

## 🔬 Methods

- **Integration**: Multi-source data merging with foreign key relationships
- **Preprocessing**: Missing value handling, standardization, feature engineering
- **Analysis**: PCA, K-means clustering, correlation analysis
- **Tools**: pandas, scikit-learn, seaborn

## 📈 Key Results

**Machine Comparison:**
- Machine 01: High variability (σ = 12.3°C temp)
- Machine 02: Most consistent (σ = 8.7°C temp)
- Machine 03: Elevated temperature (75.8°C mean)

**Operational Clusters:**
1. Normal Operation (42%)
2. High-Speed (18%)
3. Heavy Load (15%)
4. High Wear - maintenance needed (14%)
5. Low Utilization (11%)

## 📄 Files

- `index.qmd` - Publication content
- `_quarto.yml` - Configuration
- `references.bib` - Bibliography
- `README.md` - This file

## 🏗️ Building

```bash
make pdf manufacturing_analytics
```

---

**Author**: Yevheniy Chuba
**Institution**: University of Pittsburgh
**Program**: MSDS (Applied AI Emphasis)
**Date**: November 2025


