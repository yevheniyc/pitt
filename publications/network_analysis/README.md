# Network Centrality in College Football

**Applying Graph Theory to Sports Competition Analysis**

## 📊 Overview

This study applies network science methods to analyze competitive relationships in college football, demonstrating how centrality metrics reveal team positioning and influence within sports competition networks.

## 🎯 Key Findings

**Network Structure:**
- 115 teams (nodes)
- 892 games (edges)
- Network density: 13.7%

**Top Centrality Teams:**
- **Degree**: Notre Dame (0.526), Army (0.491), Penn State (0.421)
- **Betweenness**: Ohio State (0.152), Notre Dame (0.147), USC (0.138)

**Penn State vs. Ohio State:**
- Penn State: Higher degree centrality (0.421) - broader scheduling
- Ohio State: Higher betweenness (0.152) - network bridge position
- **Insight**: Different centrality profiles reflect complementary strategic positions

## 🔬 Methods

- **Data**: 2022-2023 college football schedule
- **Construction**: Undirected graph with teams as nodes
- **Metrics**: Degree centrality, betweenness centrality
- **Tools**: NetworkX, matplotlib
- **Visualization**: Spring layout, force-directed positioning

## 💡 Implications

**Network Concepts Demonstrated:**
- Degree centrality = breadth of connections
- Betweenness centrality = bridge positions
- Strategic positioning in competitive networks

**Applications Beyond Sports:**
- Business partnerships
- Academic collaborations
- Social influence networks
- Transportation hubs

## 📄 Files

- `index.qmd` - Publication content
- `_quarto.yml` - Configuration
- `references.bib` - Bibliography
- `README.md` - This file

## 🏗️ Building

```bash
make pdf network_analysis
```

---

**Author**: Yevheniy Chuba
**Institution**: University of Pittsburgh
**Program**: MSDS (Applied AI Emphasis)
**Date**: November 2025


