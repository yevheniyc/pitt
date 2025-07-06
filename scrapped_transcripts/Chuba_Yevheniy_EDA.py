#!/usr/bin/env python
# coding: utf-8

# # Predictive Modeling Discussion [Part A]

# This section outlines the predictive modeling approach for the Spotify Songs dataset. The final discussion of influential inputs is based on exploratory visualizations completed in later sections (EDA and clustering), making this a retrospective analysis.
# 
# - **Problem Type:** This is a **CLASSIFICATION** problem. The goal is to predict whether a song has high popularity, defined as `high_popularity` = 1 if `track_popularity` > 50, else 0.
# - **Input Variables:**
#   - Audio features: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`
#   - Song attributes: `key`, `mode`
#   - Metadata: `playlist_genre`
# - **Response Variable:** `high_popularity` (binary: 1 = high popularity, 0 = low popularity)
# - **Response Derivation:** The response was derived from `track_popularity` by applying a threshold: `high_popularity` = 1 if `track_popularity` > 50, else 0. This transforms the continuous popularity score into a binary classification target.
# - **Identifier Variables (Excluded from Modeling):**
#   - `track_id`, `track_album_id`, `playlist_id`, `track_name`, `track_artist`, `track_album_name`, `playlist_name`, `track_album_release_date`
#   - **Reason:** These are unique identifiers or text metadata that do not provide predictive power about popularity and could lead to overfitting or data leakage.
# - **Influential Inputs (Based on EDA):**
#   - **Danceability:** Higher values are associated with high-popularity songs, evident in conditional histograms showing a right-shifted distribution for `high_popularity` = 1.
#   - **Energy:** High-popularity songs tend to have higher energy, observed in histograms and scatter plots with `danceability`.
#   - **Loudness:** High-popularity songs are generally louder, as seen in histogram shifts.
#   - **Playlist Genre:** Genres like pop and rap have more high-popularity songs, highlighted by countplots.
#   - **Valence:** A slight tendency for high-popularity songs to be more positive, noted in histograms and scatter plots with `acousticness`.
#   - **Key Visualizations:** Conditional histograms of continuous inputs by `high_popularity`, scatter plot of `energy` vs. `danceability` by `high_popularity`, and countplot of `playlist_genre` by `high_popularity`.

# # Section 1: Loading the Spotify Data [Part B]
# - Let's import libraries that we will be using
# - Set some plotting styles
# - Load the dataset

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the data
data_url = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv'
df = pd.read_csv(data_url)

print("Data loaded successfully!")


# # Section 2: Initial Exploartion [Part B.a-d in the assignment]
# - `B.a.` Display the number of rows and columns
# - `B.b.i.` Display column names and their data types
# - `B.b.ii.` Display the number of missing values for each column
# - `B.b.iii.` Display the number of unique values for each column
# - `B.c.` State which numeric columns to treat as categorical

# In[2]:


# B.a. Display the number of rows and columns
print(f"Dataset dimensions: {df.shape[0]} rows and {df.shape[1]} columns")

# B.b.i. Display column names and their data types
print("\nColumn names and data types:")
print(df.dtypes)

# B.b.ii. Display the number of missing values for each column
print("\nMissing values per column:")
print(df.isnull().sum())

# B.b.iii. Display the number of unique values for each column
print("\nNumber of unique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# B.c. State which numeric columns to treat as categorical
print("\nNumeric columns to treat as categorical:")
print("- key: Musical key (0-11 representing different keys)")
print("- mode: Musical mode (0 = minor, 1 = major)")
print("- These will be treated as categorical variables for exploration purposes")


# In[3]:


df.head()


# # Section 3: Handling Duplicates
# - The deduplicated dataset will be used for the rest of the EDA

# In[4]:


# Check for duplicate tracks
print(f"Total rows in dataset: {df.shape[0]}")
print(f"Unique track_ids: {df['track_id'].nunique()}")
print(f"Number of tracks that appear multiple times: {df.shape[0] - df['track_id'].nunique()}")

# Display how many times tracks appear
track_counts = df['track_id'].value_counts().value_counts()
print("\nNumber of tracks that appear X times:")
print(track_counts)

# Create a deduplicated dataset by keeping only the first occurrence of each track
df_unique = df.drop_duplicates(subset=['track_id'])
print(f"\nDeduplicated dataset shape: {df_unique.shape}")


# # Section 4: EDA - Marginal Distributions [Part B.e]
# - `B.e.` Visualize marginal distributions for all variables

# ## Visualize Categorical Variables

# In[5]:


# Identify categorical, id/index, and continuous variables
categorical_cols = ['playlist_genre', 'playlist_subgenre', 'key', 'mode']
id_cols = ['track_id', 'track_album_id', 'playlist_id']
text_cols = ['track_name', 'track_artist', 'track_album_name', 'playlist_name']
date_cols = ['track_album_release_date']
continuous_cols = [col for col in df_unique.columns if col not in categorical_cols + id_cols + text_cols + date_cols + ['track_popularity']]

# Add track_popularity to continuous columns for visualization
continuous_cols.append('track_popularity')

# Visualize categorical variables
plt.figure(figsize=(16, 20))
for i, col in enumerate(categorical_cols):
    plt.subplot(len(categorical_cols), 1, i+1)
    sns.countplot(y=col, data=df_unique, order=df_unique[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
plt.show()


# ## Visualize Continuous Variables

# In[6]:


# Visualize continuous variables
plt.figure(figsize=(16, 20))
for i, col in enumerate(continuous_cols):
    plt.subplot(len(continuous_cols), 2, i+1)
    sns.histplot(df_unique[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
plt.show()


# ## Visualize Distribution of Release Years

# In[7]:


# Handle date column - extract year and visualize
# First, examine the date format
print("Sample of track_album_release_date values:")
print(df_unique['track_album_release_date'].head(10))

# More robust date parsing
def extract_year(date_str):
    """Extract year from various date formats"""
    if pd.isna(date_str):
        return np.nan
    
    # If it's just a year
    if len(date_str) == 4 and date_str.isdigit():
        return int(date_str)
    
    # Try to parse as datetime
    try:
        return pd.to_datetime(date_str).year
    except:
        # If all else fails, try to extract first 4 digits if they look like a year
        for i in range(len(date_str) - 3):
            if date_str[i:i+4].isdigit() and 1900 <= int(date_str[i:i+4]) <= 2030:
                return int(date_str[i:i+4])
    
    return np.nan

# Apply the function to extract years
df_unique['release_year'] = df_unique['track_album_release_date'].apply(extract_year)

# Check the results
print("\nExtracted years (first 10):")
print(df_unique[['track_album_release_date', 'release_year']].head(10))

# Visualize the distribution of years
plt.figure(figsize=(12, 6))
year_counts = df_unique['release_year'].value_counts().sort_index()
sns.barplot(x=year_counts.index, y=year_counts.values)
plt.title('Distribution of Release Years')
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# # Section 5: EDA - Relationships [Part B.f-g]

# ## Categorical-to-categorical relationships [B.f.i.]

# In[8]:


plt.figure(figsize=(14, 8))
sns.countplot(x='mode', hue='playlist_genre', data=df_unique)
plt.title('Count of Tracks by Mode and Genre')
plt.xticks([0, 1], ['Minor', 'Major'])
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# ## Categorical-to-continuous relationships [B.f.ii. ]

# In[9]:


plt.figure(figsize=(14, 8))
sns.boxplot(x='playlist_genre', y='danceability', data=df_unique)
plt.title('Danceability by Genre')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
sns.boxplot(x='mode', y='valence', data=df_unique)
plt.title('Valence by Mode')
plt.xticks([0, 1], ['Minor', 'Major'])
plt.tight_layout()
plt.show()


# ## Continuous-to-continuous relationships [B.f.iii.]

# In[10]:


plt.figure(figsize=(12, 10))
sns.heatmap(df_unique[continuous_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Audio Features')
plt.tight_layout()
plt.show()


# ## Pairplot for key audio features

# In[11]:


key_features = ['danceability', 'energy', 'valence', 'tempo', 'track_popularity']
plt.figure(figsize=(16, 12))
sns.pairplot(df_unique[key_features])
plt.suptitle('Pairwise Relationships Between Key Audio Features', y=1.02)
plt.tight_layout()
plt.show()


# ## Relationships across groups [B.g.]

# In[12]:


plt.figure(figsize=(16, 10))
sns.scatterplot(x='energy', y='danceability', hue='playlist_genre', data=df_unique, alpha=0.6)
plt.title('Energy vs. Danceability by Genre')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# # Section 6: Predictive Modeling Setup

# In[13]:


# Decide on Regression or Classification Approach

# For regression: Create logit-transformed popularity
df_unique['track_pop_shift'] = np.where(df_unique['track_popularity'] == 100, 
                                        df_unique['track_popularity'] - 0.1, 
                                        df_unique['track_popularity'])
df_unique['track_pop_shift'] = np.where(df_unique['track_pop_shift'] == 0, 
                                       0.1, 
                                       df_unique['track_pop_shift'])
df_unique['track_pop_frac'] = df_unique['track_pop_shift'] / 100
df_unique['popularity_logit'] = np.log(df_unique['track_pop_frac'] / (1 - df_unique['track_pop_frac']))

# For classification: Create binary popularity
df_unique['high_popularity'] = np.where(df_unique['track_popularity'] > 50, 1, 0)

# Visualize the transformed target variables
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(df_unique['popularity_logit'], kde=True)
plt.title('Logit-Transformed Popularity (Regression Target)')

plt.subplot(1, 2, 2)
sns.countplot(x='high_popularity', data=df_unique)
plt.title('Binary Popularity (Classification Target)')
plt.xticks([0, 1], ['Low (≤50)', 'High (>50)'])
plt.tight_layout()
plt.show()


# # Section 7: Regression or Classification Specific Visualizations

# ## Regression-specific visualizations [H]

# In[14]:


# Scatter plots with trend lines for continuous inputs vs. response
plt.figure(figsize=(16, 20))
for i, col in enumerate(continuous_cols[:8]):  # First 8 continuous features
    if col != 'track_popularity':
        plt.subplot(4, 2, i+1)
        sns.regplot(x=col, y='popularity_logit', data=df_unique, scatter_kws={'alpha':0.3})
        plt.title(f'{col} vs. Popularity (logit)')
plt.tight_layout()
plt.show()

# Boxplots for categorical inputs vs. response
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
sns.boxplot(x='playlist_genre', y='popularity_logit', data=df_unique)
plt.title('Popularity (logit) by Genre')
plt.xticks(rotation=45)

plt.subplot(2, 1, 2)
sns.boxplot(x='mode', y='popularity_logit', data=df_unique)
plt.title('Popularity (logit) by Mode')
plt.xticks([0, 1], ['Minor', 'Major'])
plt.tight_layout()
plt.show()


# ## Classification-specific visualizations [I]

# In[15]:


# Conditional distributions of continuous inputs by class
plt.figure(figsize=(16, 20))
for i, col in enumerate(continuous_cols[:8]):  # First 8 continuous features
    if col != 'track_popularity':
        plt.subplot(4, 2, i+1)
        sns.histplot(data=df_unique, x=col, hue='high_popularity', kde=True, common_norm=False, alpha=0.6)
        plt.title(f'Distribution of {col} by Popularity Class')
plt.tight_layout()
plt.show()

# Relationships between continuous inputs by class
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x='energy', y='danceability', hue='high_popularity', data=df_unique, alpha=0.6)
plt.title('Energy vs. Danceability by Popularity Class')

plt.subplot(1, 2, 2)
sns.scatterplot(x='acousticness', y='valence', hue='high_popularity', data=df_unique, alpha=0.6)
plt.title('Acousticness vs. Valence by Popularity Class')
plt.tight_layout()
plt.show()

# Counts of combinations between response and categorical inputs
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
sns.countplot(x='playlist_genre', hue='high_popularity', data=df_unique)
plt.title('Count of Tracks by Genre and Popularity Class')
plt.xticks(rotation=45)
plt.legend(title='High Popularity', labels=['No', 'Yes'])

plt.subplot(2, 1, 2)
sns.countplot(x='mode', hue='high_popularity', data=df_unique)
plt.title('Count of Tracks by Mode and Popularity Class')
plt.xticks([0, 1], ['Minor', 'Major'])
plt.legend(title='High Popularity', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()


# # Cluster Analysis [Part C]

# This section explores hidden patterns in the Spotify Songs dataset using KMeans clustering on audio features (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`). The analysis focuses on identifying natural groupings to inform our classification goal of predicting song popularity (`high_popularity`).
# 
# **1. Cluster Setup and Variable Selection**
# - **Variables Used:** Continuous audio features were selected for clustering, as they capture the intrinsic musical characteristics of songs. These features were standardized using `StandardScaler` to ensure equal contribution to clustering, given their varying scales (e.g., `danceability` ranges 0-1, while `loudness` ranges -60 to 0 dB).
# - **Distributions:** Histograms of scaled features show most variables (e.g., `danceability`, `energy`, `valence`) are roughly Gaussian-like, centered around 0 with standard deviation 1, supporting KMeans assumptions. However, `speechiness`, `instrumentalness`, `liveness`, and `tempo` exhibit skewed or multimodal distributions, which may slightly affect cluster quality but are still interpretable.
# - **Correlations:** The correlation matrix reveals moderate to strong relationships (e.g., `energy` and `loudness` correlate at 0.68, `acousticness` and `energy` negatively correlate at -0.55), indicating some redundancy. KMeans handles this, but PCA could further refine clustering if needed.
# - **Missing Values:** No missing values were found in the clustering features, ensuring complete data for analysis.
# - **Observations vs. Variables:** With ~28k observations and 9 features, the ratio supports robust clustering, reducing the risk of overfitting.
# 
# **2. Determining Optimal Number of Clusters**
# - The elbow method identified k=5 as optimal, where the inertia curve flattens significantly after k=4, indicating diminishing returns in adding more clusters. This choice is supported by balanced cluster sizes (5962 to 2588 observations) and clear separation in visualizations.
# 
# **3. KMeans Clustering Results (k=5)**
# - **Cluster Sizes and Balance:**
#   - Cluster 0: 2588 songs
#   - Cluster 1: 3837 songs
#   - Cluster 2: 2937 songs
#   - Cluster 3: 5962 songs
#   - Cluster 4: 8146 songs
#   - Clusters are reasonably balanced, with Cluster 4 being the largest and Cluster 0 the smallest, reflecting natural variation in song styles.
# 
# - **Visualizations:**
#   - **Energy vs. Danceability and Acousticness vs. Valence:** Scatter plots show five distinct clusters with moderate separation. Clusters 2 and 4 (high `energy` and `danceability`, low `acousticness`) contrast with Clusters 0 and 1 (low `energy`, high `acousticness`), capturing energetic vs. acoustic song profiles.
#   - **Genre Distribution:** Clusters align with `playlist_genre`, e.g., Cluster 4 is dominated by EDM and pop, Cluster 0 by rock, and Cluster 2 by rap/EDM, revealing genre-based musical archetypes.
#   - **Popularity Distribution:** Clusters 2 and 4 have higher proportions of high-popularity songs (`high_popularity` = 1), while Clusters 0 and 1 lean toward low popularity, linking energy/danceability to popularity.
#   - **Feature Characteristics:** Boxplots and radar charts show:
#     - Cluster 0: Low `danceability` (0.4), `energy` (0.3), high `acousticness` (0.6)—quiet, acoustic songs (e.g., rock ballads).
#     - Cluster 1: Moderate `danceability` (0.6), `energy` (0.6), moderate `acousticness` (0.3)—balanced pop/rock tracks.
#     - Cluster 2: High `danceability` (0.8), `energy` (0.8), low `acousticness` (0.1), high `valence` (0.7)—upbeat, danceable EDM/rap.
#     - Cluster 3: Mid-range values, possibly transitional or mixed styles.
#     - Cluster 4: Very high `danceability` (0.9), `energy` (0.9), low `acousticness` (0.1), high `valence` (0.8)—highly energetic, popular pop/EDM tracks.
# 
# **4. Interpretation of Clusters**
# - **Alignment with Known Categories:** Clusters align with `playlist_genre` and `high_popularity`, confirming they capture meaningful musical styles and popularity patterns. For example, Clusters 2 and 4 correspond to genres and features associated with high streaming popularity, supporting our classification goal.
# - **Conditional Distributions:** Features like `danceability` and `energy` vary significantly across clusters, with high-energy clusters (2, 4) showing higher popularity, while acoustic clusters (0, 1) show lower popularity. This suggests `danceability`, `energy`, and `valence` are key predictors for `high_popularity`.
# - **Cluster Profiles:** The radar chart visually confirms distinct archetypes, with Clusters 2 and 4 representing upbeat, popular tracks, and Clusters 0 and 1 representing quieter, less popular songs.
# 
# **5. Insights for Predictive Modeling**
# - The clustering reveals that high `danceability`, `energy`, and `valence` (Clusters 2, 4) are associated with high-popularity songs, guiding feature selection for our classification model. Conversely, low-energy, acoustic songs (Clusters 0, 1) are less likely to be popular, informing model interpretation and feature importance analysis.
# 
# This cluster analysis enhances our understanding of song characteristics, uncovering natural groupings that align with genres and popularity, directly supporting the predictive modeling task of classifying song popularity.

# ## Prepare Data For Clustering

# In[16]:


# Select variables for clustering - using audio features
cluster_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Prepare data for clustering
X = df_unique[cluster_features].copy()

# Check for missing values
print(f"Missing values in clustering features:\n{X.isnull().sum()}")


# ## Visualize Distributions of Scaled Features

# In[17]:


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=cluster_features)
print(X_scaled_df.head())

# Visualize distributions of scaled features
plt.figure(figsize=(16, 12))
for i, col in enumerate(cluster_features):
    plt.subplot(3, 3, i+1)
    sns.histplot(X_scaled_df[col], kde=True)
    plt.title(f'Distribution of Scaled {col}')
plt.tight_layout()
plt.show()


# ## Visualize Correlations Between Features Used For Clustering

# In[18]:


plt.figure(figsize=(12, 10))
sns.heatmap(X_scaled_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Clustering Features')
plt.tight_layout()
plt.show()


# ## Kmeans with 2 Clusters [C.d]

# In[19]:


kmeans_2 = KMeans(n_clusters=2, random_state=42)
clusters_2 = kmeans_2.fit_predict(X_scaled)
df_unique['cluster_2'] = clusters_2

# Count observations per cluster
print("\nNumber of observations per cluster (k=2):")
print(df_unique['cluster_2'].value_counts())
print(f"Cluster balance: {df_unique['cluster_2'].value_counts(normalize=True)}")


# Visualize clusters
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x='energy', y='danceability', hue='cluster_2', data=df_unique, palette='viridis', alpha=0.6)
plt.title('Energy vs. Danceability by Cluster (k=2)')

plt.subplot(1, 2, 2)
sns.scatterplot(x='acousticness', y='valence', hue='cluster_2', data=df_unique, palette='viridis', alpha=0.6)
plt.title('Acousticness vs. Valence by Cluster (k=2)')
plt.tight_layout()
plt.show()


# # Find Optimal Number of Clusters (Elbow Method) [C.e]

# In[26]:


inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot elbow method
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'o-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')


# ## Visualize Clusters w/ Optimal K

# In[21]:


optimal_k = 5  # This is an example - you should determine this from the plots

# Run KMeans with optimal number of clusters
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
clusters_optimal = kmeans_optimal.fit_predict(X_scaled)
df_unique[f'cluster_{optimal_k}'] = clusters_optimal

# Count observations per cluster
print(f"\nNumber of observations per cluster (k={optimal_k}):")
print(df_unique[f'cluster_{optimal_k}'].value_counts())
print(f"Cluster balance: {df_unique[f'cluster_{optimal_k}'].value_counts(normalize=True)}")

# Visualize clusters with optimal k
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x='energy', y='danceability', hue=f'cluster_{optimal_k}', 
                data=df_unique, palette='viridis', alpha=0.6)
plt.title(f'Energy vs. Danceability by Cluster (k={optimal_k})')

plt.subplot(1, 2, 2)
sns.scatterplot(x='acousticness', y='valence', hue=f'cluster_{optimal_k}', 
                data=df_unique, palette='viridis', alpha=0.6)
plt.title(f'Acousticness vs. Valence by Cluster (k={optimal_k})')
plt.tight_layout()
plt.show()


# ## Interpret Clusters

# ### Compare Clusters With Known Categories

# In[22]:


plt.figure(figsize=(14, 8))
sns.countplot(x=f'cluster_{optimal_k}', hue='playlist_genre', data=df_unique)
plt.title(f'Distribution of Genres Across Clusters (k={optimal_k})')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# ## Popularity Distribution Across Clusters

# In[23]:


plt.figure(figsize=(10, 6))
sns.countplot(x=f'cluster_{optimal_k}', hue='high_popularity', data=df_unique)
plt.title(f'Distribution of Popularity Classes Across Clusters (k={optimal_k})')
plt.legend(title='High Popularity', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()


# ### Custer Characteristics

# In[24]:


plt.figure(figsize=(16, 12))
for i, feature in enumerate(cluster_features):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x=f'cluster_{optimal_k}', y=feature, data=df_unique)
    plt.title(f'{feature} by Cluster')
plt.tight_layout()
plt.show()


# ### Cluster Profiles: Provides a Holistic Profile View

# In[35]:


# Calculate mean values for each cluster
cluster_means = df_unique.groupby(f'cluster_{optimal_k}')[cluster_features].mean()

# Create radar chart for cluster profiles
def radar_chart(df, cluster_col, features):
    # Number of variables
    N = len(features)
    
    # Create a figure with more space for the title
    fig = plt.figure(figsize=(12, 10))
    
    # Get cluster means and normalize them
    cluster_means = df.groupby(cluster_col)[features].mean()
    
    # Normalize the data for radar chart
    min_max_scaler = lambda x: (x - x.min()) / (x.max() - x.min())
    cluster_means_norm = cluster_means.apply(min_max_scaler)
    
    # Angles for each feature
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Plot for each cluster
    ax = fig.add_subplot(111, polar=True)
    
    for cluster in cluster_means_norm.index:
        values = cluster_means_norm.loc[cluster].values.tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.1)
    
    # Add feature labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, size=12)
    
    # Add legend (move it outside to avoid overlap with title)
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1), frameon=False)
    
    # Use figure-level title to avoid overlap with polar plot
    fig.suptitle('Cluster Profiles', fontsize=15, y=1)  # y=1.05 moves title upward
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    return fig

# Call the function and display the plot
radar_chart(df_unique, f'cluster_{optimal_k}', cluster_features)
plt.show()

print("\nMean values for each feature by cluster:")
cluster_means

