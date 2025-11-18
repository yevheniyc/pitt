# Data-Driven Fire Safety Analytics

**Leveraging 930,000 Emergency Dispatch Records for Public Safety Intelligence**

## 📊 Overview

This applied analytics research examines 10 years of emergency fire dispatch data from Allegheny County, Pennsylvania, to identify patterns that inform resource allocation and policy decisions. The study combines descriptive analytics, geospatial visualization, and interactive dashboard development to make public safety intelligence accessible to stakeholders.

## 🎯 Key Findings

- **False Alarm Crisis**: 37.3% of dispatches (347,191) were fire alarms, with ~65% estimated to be false ($225.7M cost over 10 years)
- **Geographic Disparities**: Some municipalities show 3× higher per-capita incident rates
- **Seasonal Patterns**:
  - Structure fires peak in winter (+34% during Dec-Feb)
  - Outdoor fires peak in summer (+78% during Jun-Aug)
  - Fire alarms consistent year-round (systemic issue)
- **Hourly Patterns**: Bimodal distribution with morning (8-11 AM) and evening (5-8 PM) peaks

## 🔬 Methods

- **Dataset**: 930,808 dispatch records (2015-2024) from WPRDC
- **Analysis**: Descriptive statistics, geospatial analysis, temporal decomposition
- **Tools**: Python (pandas, geopandas, plotly, folium)
- **Dashboard**: Interactive Gradio web application

## 📈 Results Summary

### Volume Distribution

| Incident Type | Count | Percentage |
|--------------|-------|------------|
| Fire Alarms | 347,191 | 37.3% |
| Medical Assists | 186,524 | 20.0% |
| Structure Fires | 89,342 | 9.6% |
| Vehicle Fires | 52,187 | 5.6% |
| Other | 255,564 | 27.5% |

### Geographic Hotspots (Per-Capita Rates)

1. Homestead: 47.2 per 1,000 residents
2. Braddock: 43.8 per 1,000 residents
3. Rankin: 41.5 per 1,000 residents
4. Duquesne: 39.7 per 1,000 residents
5. Wilkinsburg: 38.2 per 1,000 residents

## 💡 Policy Recommendations

**1. Smart Alarm Technology Deployment**
- Target: 30-50% false alarm reduction
- Focus: Top 100 high-frequency addresses
- Expected savings: $67-112M over 10 years

**2. Targeted Community Prevention**
- Target: 20% fire reduction in top 5 municipalities
- Strategy: Door-to-door smoke alarms, safety education
- Impact: 3,000+ fewer structure fires over 10 years

**3. Seasonal Resource Reallocation**
- Target: Align staffing with demand patterns
- Strategy: Increase capacity during peak seasons/hours
- Impact: Reduced response times, efficient resource use

## 🛠️ Technical Stack

- **Data Processing**: pandas, numpy
- **Geospatial**: geopandas, folium, shapely
- **Visualization**: plotly, matplotlib, seaborn
- **Dashboard**: Gradio
- **Publication**: Quarto PDF generation

## 📄 Files

- `index.qmd` - Main publication content (35+ pages)
- `_quarto.yml` - Publication configuration
- `references.bib` - Bibliography
- `README.md` - This file

## 🏗️ Building

```bash
# From pitt/ directory
make pdf fire_safety_dashboard

# Output: publications/pdf/Data-Driven-Fire-Safety-Analytics.pdf
```

## 📊 Dashboard Features

The interactive dashboard provides:

- **Geographic Analysis**: Heat maps and municipality comparisons
- **Temporal Trends**: Year-over-year and seasonal patterns
- **Emergency Priorities**: Distribution and response insights
- **False Alarm Analysis**: Cost impact calculations

## 🎓 Research Context

This work was completed as part of the Data Visualization course at University of Pittsburgh, demonstrating practical application of analytics to real-world public safety challenges.

## 📖 Citation

```bibtex
@techreport{chuba2025firesafety,
  title = {Data-Driven Fire Safety Analytics: Leveraging 930,000 Emergency Dispatch Records},
  author = {Chuba, Yevheniy},
  institution = {University of Pittsburgh},
  year = {2025},
  type = {Applied Analytics Research Publication}
}
```

## 🔗 Related Resources

- **Data Source**: [Western Pennsylvania Regional Data Center](https://data.wprdc.org)
- **NFPA Research**: [National Fire Protection Association](https://www.nfpa.org/research)
- **Original Dashboard**: Gradio application (local deployment)

---

**Author**: Yevheniy Chuba
**Institution**: University of Pittsburgh
**Program**: MSDS (Applied AI Emphasis)
**Course**: Data Visualization
**Date**: November 2025


