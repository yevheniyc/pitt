# Predicting Song Popularity on Spotify

**A Machine Learning Approach to Understanding Audio Features and Genre Effects**

## 📊 Overview

This research publication investigates the predictability of song popularity on Spotify using logistic regression and interaction modeling. The study analyzes audio features across six genres to understand what makes songs succeed on streaming platforms.

## 🎯 Key Findings

- **ROC AUC**: 0.675 (robust cross-validation)
- **Primary Predictor**: Genre classification (pop and rock 1.97× and 1.72× higher odds)
- **Critical Audio Features**:
  - Instrumentalness: -0.22 (strong negative)
  - Loudness: +0.29 (professional production)
  - Danceability: Genre-dependent (critical for rap +0.33)
- **Genre Interactions**: Audio feature effects vary substantially by genre

## 🔬 Methods

- **Dataset**: Spotify songs with 14 audio features across 6 genres
- **Approach**: EDA → Clustering → Progressive logistic regression models
- **Validation**: Stratified 5-fold cross-validation
- **Final Model**: Genre + Audio features + Interactions (Model 6)

## 📈 Results Highlights

1. **Genre dominates**: Accounts for 62.5% baseline performance
2. **Audio features add value**: Improve ROC AUC by 2.3 percentage points
3. **Interactions are crucial**: Final 4% improvement from genre×audio terms
4. **Robust generalization**: Training vs. CV AUC difference < 0.004

## 💡 Practical Implications

**For Artists/Producers:**
- Vocal-forward tracks generally more popular
- Genre conventions matter more than absolute audio values
- Dance appeal critical for rap and pop, less so for rock

**For Platforms:**
- Genre-aware recommendations validated by interaction effects
- Audio similarity insufficient without genre context

## 🛠️ Technical Stack

- **Analysis**: Python 3.8+, pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Modeling**: Logistic regression with interactions
- **Publication**: Quarto PDF generation

## 📄 Files

- `index.qmd` - Main publication content (Quarto markdown)
- `_quarto.yml` - Publication configuration
- `references.bib` - Bibliography
- `README.md` - This file

## 🏗️ Building

```bash
# From pitt/ directory
make pdf spotify_popularity

# Output: publications/pdf/Predicting-Song-Popularity-on-Spotify.pdf
```

## 📚 Citation

```bibtex
@techreport{chuba2025spotify,
  title = {Predicting Song Popularity on Spotify: A Machine Learning Approach},
  author = {Chuba, Yevheniy},
  institution = {University of Pittsburgh},
  year = {2025},
  type = {Academic Research Publication}
}
```

## 📖 Related Work

This research builds on:
- Million Song Dataset research [@bertin2011]
- Social influence in cultural markets [@salganik2006]
- Audio feature analysis in music information retrieval

---

**Author**: Yevheniy Chuba
**Institution**: University of Pittsburgh
**Program**: MSDS (Applied AI Emphasis)
**Course**: Introduction to Data Science
**Date**: November 2025


