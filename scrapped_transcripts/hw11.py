#!/usr/bin/env python
# coding: utf-8

# ## CMPINF 2100: Homework 11
# 
# ## Overview
# 
# This assignment is focused on interpreting predictive models. The first 4 problems are REGRESSION problems and the last problem is a BINARY CLASSIFICATION problem. The prevailing theme through the assignment is that you will practice working with various features derived from inputs. You will study how these features impact model behavior by examining coefficient summaries and visualizing predictions. The REGRESSION problems involve 2 inputs to give you experience working with ADDITIVE and INTERACTION features. You will also gain experience with CATEGORICAL or NON-NUMERIC inputs in the REGRESSION problems. The BINARY CLASSIFICATION problem involves a single input.
# 
# **You must download the 4 data sets provided in the Canvas assignment page and save them to the appropriate directory on your computer.**
# 
# The 4 REGRESSION problems use the same column naming convention. The inputs are named `x1` and `x2` while the output (response) is named `y`. Please pay close attention to the DATA TYPE of the inputs in the different problems. The BINARY CLASSICIATION problem uses its own column naming convention. The input is named `x` and the binary outcome (output) is named `y`. Pay close attention to the data type of the columns in the BINARY CLASSIFICATION problem!  
# 
# You will NOT perform detailed visual data exploration as part of this assignment. Please note this is **not** because Exploratory Data Analysis (EDA) is not an important aspect of modeling. You should ALWAYS perform EDA **before** modeling. EDA is mostly skipped here because the assignment is focused on fitting and interpreting models. That said, certain problems involve exploratory visualizations to help address important concepts. 
# 
# ## Problem 00
# 
# ### 00a)
# 
# You will work with the "big 4" modules of NumPy, Pandas, matplotlib.pyplot, and Seaborn in this assignment.   
# 
# Import NumPy, Pandas, matplotlib.pyplot, and Seaborn using their commonly accepted aliases.
# 
# #### 00a) - SOLUTION

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# ### 00b)
# 
# You will use the statsmodels formula interface to fit the models in all problems. Import the statsmodels.formula.api using its common alias.
# 
# **ALL** models that you fit in this assignment will use the statsmodels formula interface!  
# 
# #### 00b) - SOLUTION

# In[2]:


import statsmodels.formula.api as smf


# ## Problem 01
# 
# This is a REGRESSION problem involving 2 inputs, `x1` and `x2`. The output is named `y`. 
# 
# ### 1a)
# 
# Read in the `hw11_probA.csv` CSV file and assign it to the `dfA` object.
# 
# Apply the `.info()` method to display useful information associated with the columns in the data set.
# 
# #### 1a) - SOLUTION

# In[4]:


# Read in the dataset and store it in dfA
dfA = pd.read_csv('hw11_probA.csv')

# Display information about the dataframe
dfA.info()


# ### 1b) 
# 
# Visualize the relationship between the output `y` and each input using a TREND PLOT. You may create 2 separate figures OR you may reshape the data to LONG-FORMAT to visualize the TREND PLOTS associated with each input via FACETS.
# 
# Regardless of your approach for creating the figures, how would you describe the RELATIONSHIP between the output `y` and each input based on the figure?
# 
# #### 1b) - SOLUTION

# In[7]:


# Approach 1: Two separate figures
plt.figure(figsize=(12, 5))

# Plot for x1 vs y
plt.subplot(1, 2, 1)
sns.regplot(x='x1', y='y', data=dfA, scatter_kws={'alpha':0.5})
plt.title('Relationship between x1 and y')

# Plot for x2 vs y
plt.subplot(1, 2, 2)
sns.regplot(x='x2', y='y', data=dfA, scatter_kws={'alpha':0.5})
plt.title('Relationship between x2 and y')

plt.tight_layout()
plt.show()


# #### Observation:
# Both x1 and x2 show weak negative relationships with y, with substantial scatter around the regression lines and wide confidence intervals. The similar patterns suggest potential collinearity between predictors. This indicates we should include both variables in our model, consider interaction effects, and expect moderate rather than strong predictive performance.

# ### 1c)
# 
# You will fit many different models in this assignment. It will be helpful to visualize the coefficient summaries with a COEFFICIENT PLOT rather than printing information to the screen.
# 
# Previous lecture recordings demonstrated how to create a COEFFICIENT PLOT by defining a function `my_coefplot()`. Define the `my_coefplot()` function in the cell below follwing the example from lecture. You may use the 95% confidence interval approximation based on multiplying 2 times the standard error.
# 
# #### 1c) - SOLUTION

# In[8]:


def my_coefplot(model_results, title=None, figsize=(10, 6)):
    """
    Create a coefficient plot with error bars for a fitted statsmodels model.
    
    Parameters:
    -----------
    model_results : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted model results object
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Extract the coefficient table
    coef_df = pd.DataFrame({
        'coef': model_results.params,
        'stderr': model_results.bse,
        'pvalue': model_results.pvalues
    })
    
    # Reset the index to make the variable names a column
    coef_df = coef_df.reset_index()
    coef_df = coef_df.rename(columns={'index': 'variable'})
    
    # Compute the 95% confidence intervals (approximately 2 standard errors)
    coef_df['ci_lower'] = coef_df['coef'] - 2 * coef_df['stderr']
    coef_df['ci_upper'] = coef_df['coef'] + 2 * coef_df['stderr']
    
    # Exclude the intercept if it exists
    if 'Intercept' in coef_df['variable'].values:
        plot_df = coef_df[coef_df['variable'] != 'Intercept'].copy()
    else:
        plot_df = coef_df.copy()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the coefficients as points
    ax.scatter(plot_df['coef'], plot_df['variable'], s=80, color='blue')
    
    # Add error bars for confidence intervals
    for i, row in plot_df.iterrows():
        ax.plot([row['ci_lower'], row['ci_upper']], [row['variable'], row['variable']], 
                color='blue', alpha=0.7)
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Coefficient Value')
    ax.set_ylabel('Variable')
    if title:
        ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax


# ### 1d)
# 
# Let's start fitting models to predict the output given the inputs. You will begin by fitting models using a SINGLE input at a time to mirror TREND PLOTS you created in the previous problem.
# 
# Fit a linear model that assumes a LINEAR relationship between the AVERAGE OUTPUT (TREND) and `x1` only. Assign the fitted model to the `fit_A_x1` object. 
# 
# Display the COEFFICIENT PLOT associated with `fit_A_x1`.
# 
# Are you confident in the relationship between the TREND and `x1`?  
# 
# #### 1d) - SOLUTION

# In[9]:


# Fit a linear model with x1 as the only predictor
fit_A_x1 = smf.ols(formula='y ~ x1', data=dfA).fit()

# Display the coefficient plot
fig, ax = my_coefplot(fit_A_x1, title='Coefficients for Model with x1 Only')
plt.show()


# In[ ]:





# #### Observation
# 
# I am not confident in the relationship between the trend and x1. The confidence interval for the x1 coefficient crosses zero (the red dashed line), which indicates that the relationship is not statistically significant at the 95% confidence level. While the point estimate suggests a negative relationship (approximately -0.3), we cannot rule out that the true relationship might be zero or even slightly positive.
# 
# This confirms what we observed in the scatter plot - a weak relationship with substantial uncertainty. The evidence is insufficient to confidently claim that x1 has a meaningful effect on y.

# ### 1e)
# 
# Fit a linear model that assumes a LINEAR relationship between the AVERAGE OUTPUT (TREND) and `x2` only. Assign the fitted model to the `fit_A_x2` object. 
# 
# Display the COEFFICIENT PLOT associated with `fit_A_x2`.
# 
# Are you confident in the relationship between the TREND and `x2`? 
# 
# #### 1e) - SOLUTION

# In[10]:


# Fit a linear model with x2 as the only predictor
fit_A_x2 = smf.ols(formula='y ~ x2', data=dfA).fit()

# Display the coefficient plot
fig, ax = my_coefplot(fit_A_x2, title='Coefficients for Model with x2 Only')
plt.show()


# In[ ]:





# ### Observation
# 
# Looking at the coefficient plot for x2, I am not confident in the relationship between the trend and x2.
# 
# Similar to what we observed with x1, the coefficient for x2 is negative (approximately -0.35), but the confidence interval appears to cross zero (the red dashed vertical line). The blue horizontal line representing the 95% confidence interval extends from about -0.7 on the left to slightly past 0 on the right.
# 
# Since the confidence interval includes zero, we cannot reject the null hypothesis that the true effect of x2 on y might be zero. This suggests that while there's a suggested negative relationship, it's not statistically significant at the 95% confidence level.
# 
# This result aligns with our initial visualization where we observed a weak negative relationship with substantial scatter. Neither x1 nor x2 individually shows a strong, significant relationship with y.

# ### 1f)
# 
# Let's now fit a model that uses BOTH inputs! Fit a linear model that assumes LINEAR ADDITIVE FEATURES associated with BOTH inputs. Assign the fitted model to the `fit_A_add` object.
# 
# Display the COEFFICIENT PLOT associated with `fit_A_add`.
# 
# Are you confident in the relationships between the TREND and either input?
# 
# #### 1f) - SOLUTION

# In[12]:


# Fit a linear model with both x1 and x2 as additive predictors
fit_A_add = smf.ols(formula='y ~ x1 + x2', data=dfA).fit()

# Display the coefficient plot
fig, ax = my_coefplot(fit_A_add, title='Coefficients for Additive Model with x1 and x2')
plt.show()


# In[ ]:





# ### Observation:
# Based on the coefficient plot for the additive model with both x1 and x2, I'm still not confident in the relationships between the trend and either input.
# 
# For both x1 and x2:
# - The coefficient estimates are negative (both around -0.3)
# - The 95% confidence intervals (blue horizontal lines) for both coefficients cross the zero line (red dashed vertical line)
# - This means neither relationship is statistically significant at the 95% confidence level
# 
# Even when including both predictors in the model simultaneously, neither x1 nor x2 shows a significant relationship with y. The confidence intervals for both variables include zero, suggesting that we cannot rule out the possibility that the true effect of either variable might be zero.
# 
# This result is consistent with what we observed in the individual models. The combination of both predictors in an additive model doesn't substantially change the conclusions - there is no strong evidence for a significant relationship between the trend and either input variable.

# ### 1g)
# 
# Let's now include an INTERACTION between the TWO inputs. Fit a linear model that involves the LINEAR **main effects** AND the INTERACTION between BOTH inputs. Assign the fitted model to the `fit_A_int` object.
# 
# Display the COEFFICIENT PLOT associated with `fit_A_int`.
# 
# Are you confident in the relationships between the TREND and any of the FEATURES included in the model?
# 
# #### 1g) - SOLUTION

# In[13]:


# Fit a linear model with both main effects and interaction term
fit_A_int = smf.ols(formula='y ~ x1 + x2 + x1:x2', data=dfA).fit()

# Display the coefficient plot
fig, ax = my_coefplot(fit_A_int, title='Coefficients for Model with Main Effects and Interaction')
plt.show()


# In[ ]:





# Based on the coefficient plot for the model with main effects and interaction, I am confident in the relationships between the trend and two of the features:
# 
# 1. **x2 (main effect)**: The coefficient is approximately -0.5, and its confidence interval is entirely to the left of zero, not crossing the red dashed line. This indicates a statistically significant negative relationship between x2 and y.
# 
# 2. **x1:x2 (interaction)**: The coefficient is approximately +2.3, with a confidence interval entirely to the right of zero. This indicates a strong, statistically significant positive interaction effect between x1 and x2.
# 
# 3. **x1 (main effect)**: The coefficient is close to zero (slightly positive), and its confidence interval crosses the red dashed line at zero. This suggests that the main effect of x1 is not statistically significant.
# 
# This is a dramatic change from the previous models! The inclusion of the interaction term reveals important dynamics:
# 
# - The effect of x2 is significantly negative when x1 = 0
# - The interaction term (x1:x2) indicates that the effect of x2 changes (becomes more positive) as x1 increases
# - Similarly, the effect of x1 depends strongly on the value of x2
# 
# This suggests that neither x1 nor x2 alone were good predictors, but their interaction is crucial for understanding the relationship with y. The strong interaction explains why we didn't see significant effects in the additive model - the relationship is more complex than simple additive effects.
# 

# ### 1h)
# 
# You have fit 4 linear models in this problem.
# 
# Which of the 4 models has the BEST R-squared on the TRAINING data?
# 
# #### 1h) - SOLUTION

# In[14]:


# Extract R-squared values for all models
models = {
    'x1 only': fit_A_x1,
    'x2 only': fit_A_x2,
    'Additive (x1 + x2)': fit_A_add,
    'Interaction (x1 + x2 + x1:x2)': fit_A_int
}

# Create a dataframe to compare R-squared values
rsquared_df = pd.DataFrame({
    'Model': list(models.keys()),
    'R-squared': [model.rsquared for model in models.values()]
}).sort_values('R-squared', ascending=False)

# Display the results
print(rsquared_df)

# Identify the best model
best_model = rsquared_df.iloc[0]['Model']
best_rsquared = rsquared_df.iloc[0]['R-squared']

print(f"\nThe model with the best R-squared is: {best_model} with R-squared = {best_rsquared:.4f}")


# ## Problem 02
# 
# This problem continues working with the `dfA` data set. You fit 4 models previously, and focused on the behavior through the estimated coefficients. You will continue interpreting the model behavior in this problem but you will do so through predictions. You will only work with the model that included the **main effects** AND INTERACTION between BOTH inputs in this problem.
# 
# You visualized model predictions in the previous assignment. To do so, you needed to create a NEW data set which had values to support the predictive visualizations. You will create NEW data sets for this assignment to support visualilzations, BUT these NEW data sets MUST include values for BOTH inputs! This is because the model you working with, `fit_A_int`, has features derived from BOTH inputs. Predictions **cannot** be made if BOTH inputs are **not** present!
# 
# ### 2a)
# 
# You will begin by making predictions that include the predictive TREND, the UNCERTAINTY on the TREND (the confidence interval), and the UNCERTAINTY on a single measurement (the prediction interval). 
# 
# You will focus on the relationship between the output and `x2`. Therefore, `x2` will be the "primary" input in the NEW data set and have the most unqiue values. The `x1` input must be set to a CONSTANT value so that way you will visualize a SINGLE set of RIBBONS for the UNCERTAINTY INTERVALS. Inputs held constant are typically set to the TRAINING set MEDIAN or AVERAGE in order to capture the behavior "near the center". 
# 
# However, for this assignment you will visualize the predictive behavior with `x1` set to a value away from the CENTER. You will use a value that is HALF WAY between the TRAINING set MAXIMUM and the TRAINING set AVERAGE for `x1`.
# 
# Create a NEW Pandas DataFrame, `dfA_viz_a`, that contains a two columns named `x1` and `x2`. The `x2` column must consist of 101 evenly spaced values between the TRAINING set MINIMUM and TRAINING set MAXIMUM `x2` values. The `x1` column must be a CONSTANT value equal to the MIDPOINT (HALF WAY) between the TRAINING set MAXIMUM and the TRAINING set AVERAGE for `x1`. Remember that the training set is contained in the `dfA` DataFrame.
# 
# Display the `.nunique()` method associated with `dfA_viz_a` to the screen to confirm it was created correctly.
# 
# #### 2a) - SOLUTION

# In[15]:


# Calculate statistics from the training set
x1_avg = dfA['x1'].mean()
x1_max = dfA['x1'].max()
x2_min = dfA['x2'].min()
x2_max = dfA['x2'].max()

# Calculate the value halfway between the max and average for x1
x1_constant = x1_avg + (x1_max - x1_avg) / 2

# Create a new DataFrame with 101 evenly spaced points for x2
x2_values = np.linspace(x2_min, x2_max, 101)
dfA_viz_a = pd.DataFrame({
    'x1': [x1_constant] * 101,  # Constant value for x1
    'x2': x2_values  # 101 evenly spaced values for x2
})

# Display the number of unique values in each column
print(dfA_viz_a.nunique())


# In[ ]:





# ### 2b)
# 
# Perform the necessary actions to **SUMMARIZE** the predictions associated with `fit_A_int` on the NEW `dfA_viz_a` data set. The predictions must include the predicted MEAN output (the TREND), the confidence interval bounds, and the prediction interval bounds.
# 
# Assign the prediction summaries to the `fit_A_pred_summary_a` object.
# 
# Display the head of `fit_A_pred_summary_a` object to the screen to confirm it is created correctly.
# 
# #### 2b) - SOLUTION

# In[17]:


# Generate predictions with confidence and prediction intervals
fit_A_pred_a = fit_A_int.get_prediction(dfA_viz_a)
fit_A_pred_summary_a = fit_A_pred_a.summary_frame(alpha=0.05)

# Display the head of the prediction summary DataFrame
fit_A_pred_summary_a.head()


# ### 2c)
# 
# You now have everything necessary to visualize the predictions of the `fit_A_int` model on the NEW data, `dfA_viz_a`! You must visualize the predictive trend and BOTH types of uncertainty with respect to the input `x2`.
# 
# You must visualize the predicted MEAN output (the trend) as a line. You must visualize the confidence interval as a grey ribbon. You must visualize the prediction interval as an orange ribbon. You do NOT need to include the training set for this problem.
# 
# Create the figure using matplotlib methods associated with the matplotlib axis object. Label the x and y axis correctly.
# 
# #### 2c) - SOLUTION

# In[18]:


# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the prediction interval (orange ribbon)
ax.fill_between(dfA_viz_a['x2'], 
                fit_A_pred_summary_a['obs_ci_lower'],
                fit_A_pred_summary_a['obs_ci_upper'],
                color='orange', alpha=0.3, label='Prediction Interval (95%)')

# Plot the confidence interval (grey ribbon)
ax.fill_between(dfA_viz_a['x2'], 
                fit_A_pred_summary_a['mean_ci_lower'],
                fit_A_pred_summary_a['mean_ci_upper'],
                color='grey', alpha=0.5, label='Confidence Interval (95%)')

# Plot the mean prediction (trend line)
ax.plot(dfA_viz_a['x2'], fit_A_pred_summary_a['mean'], 
        color='blue', linewidth=2, label='Predicted Mean')

# Add labels and title
ax.set_xlabel('x2', fontsize=12)
ax.set_ylabel('y', fontsize=12)
x1_value = dfA_viz_a['x1'].iloc[0]  # Get the constant x1 value
title = f'Predictions vs x2 (with x1 fixed at {x1_value:.2f})'
ax.set_title(title, fontsize=14)

# Add legend
ax.legend(loc='best')

# Display grid
ax.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and display
plt.tight_layout()
plt.show()


# ### 2d)
# 
# The predictions visualized in 2c) consider a wide range of `x2` values **but** are specific to a SINGLE `x1` value! Let's see what happens if you CHANGE `x1` to a different CONSTANT value! The new value of `x1` will be between the MINIMUM and AVERAGE instead of the MAXIMUM and AVERAGE.
# 
# Create a NEW Pandas DataFrame, `dfA_viz_b`, that contains a two columns named `x1` and `x2`. The `x2` column must consist of 101 evenly spaced values between the TRAINING set MINIMUM and TRAINING set MAXIMUM `x2` values. The `x1` column must be a CONSTANT value equal to the MIDPOINT (HALF WAY) between the TRAINING set MINIMUM and the TRAINING set AVERAGE for `x1`. Remember that the training set is contained in the `dfA` DataFrame.
# 
# Display the `.nunique()` method associated with `dfA_viz_b` to the screen to confirm it was created correctly.
# 
# #### 2d) - SOLUTION

# In[19]:


# Calculate statistics from the training set
x1_avg = dfA['x1'].mean()
x1_min = dfA['x1'].min()
x2_min = dfA['x2'].min()
x2_max = dfA['x2'].max()

# Calculate the value halfway between the min and average for x1
x1_constant = x1_min + (x1_avg - x1_min) / 2

# Create a new DataFrame with 101 evenly spaced points for x2
x2_values = np.linspace(x2_min, x2_max, 101)
dfA_viz_b = pd.DataFrame({
    'x1': [x1_constant] * 101,  # Constant value for x1
    'x2': x2_values  # 101 evenly spaced values for x2
})

# Display the number of unique values in each column
print(dfA_viz_b.nunique())


# In[ ]:





# ### 2e)
# 
# Perform the necessary actions to **SUMMARIZE** the predictions associated with `fit_A_int` on the second NEW `dfA_viz_b` data set. The predictions must include the predicted MEAN output (the TREND), the confidence interval bounds, and the prediction interval bounds.
# 
# Assign the prediction summaries to the `fit_A_pred_summary_b` object.
# 
# Display the head of `fit_A_pred_summary_b` object to the screen to confirm it is created correctly.
# 
# #### 2e) - SOLUTION

# In[21]:


# Generate predictions with confidence and prediction intervals for the second dataset
fit_A_pred_b = fit_A_int.get_prediction(dfA_viz_b)
fit_A_pred_summary_b = fit_A_pred_b.summary_frame(alpha=0.05)

# Display the head of the prediction summary DataFrame
fit_A_pred_summary_b.head()


# ### 2f)
# 
# You now have everything necessary to visualize the predictions of the `fit_A_int` model on the second NEW data, `dfA_viz_b`! You must visualize the predictive trend and BOTH types of uncertainty with respect to the input `x2`.
# 
# You must visualize the predicted MEAN output (the trend) as a line. You must visualize the confidence interval as a grey ribbon. You must visualize the prediction interval as an orange ribbon. You do NOT need to include the training set for this problem.
# 
# Create the figure using matplotlib methods associated with the matplotlib axis object. Label the x and y axis correctly.
# 
# #### 2f) - SOLUTION

# In[22]:


# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the prediction interval (orange ribbon)
ax.fill_between(dfA_viz_b['x2'], 
                fit_A_pred_summary_b['obs_ci_lower'],
                fit_A_pred_summary_b['obs_ci_upper'],
                color='orange', alpha=0.3, label='Prediction Interval (95%)')

# Plot the confidence interval (grey ribbon)
ax.fill_between(dfA_viz_b['x2'], 
                fit_A_pred_summary_b['mean_ci_lower'],
                fit_A_pred_summary_b['mean_ci_upper'],
                color='grey', alpha=0.5, label='Confidence Interval (95%)')

# Plot the mean prediction (trend line)
ax.plot(dfA_viz_b['x2'], fit_A_pred_summary_b['mean'], 
        color='blue', linewidth=2, label='Predicted Mean')

# Add labels and title
ax.set_xlabel('x2', fontsize=12)
ax.set_ylabel('y', fontsize=12)
x1_value = dfA_viz_b['x1'].iloc[0]  # Get the constant x1 value
title = f'Predictions vs x2 (with x1 fixed at {x1_value:.2f})'
ax.set_title(title, fontsize=14)

# Add legend
ax.legend(loc='best')

# Display grid
ax.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and display
plt.tight_layout()
plt.show()


# ### 2g)
# 
# The previous figures focused on a SINGLE `x1` value because BOTH types of UNCERTAINTY intervals were included. However, let's now directly examine how `x1` impacts the relationship between the TREND and `x2` by including multiple values of `x1` in the predictions. To do so, you will need to create a third NEW data set that includes multiple values of `x2` AND multiple values of `x1`. The `x2` input will continue to be the primary input and thus have the most unique values.
# 
# Create a NEW Pandas DataFrame, `dfA_viz_c`, that contains a two columns named `x1` and `x2`. The `x2` column must consist of 101 evenly spaced values between the TRAINING set MINIMUM and training set MAXIMUM `x2` values. The `x1` column consist of 9 evenly spaced values between the TRAINING set MINIMUM and TRAINING set MAXIMUM `x1` values. Thus, `x1` is NOT constant! Remember that the training set is contained in the `dfA` DataFrame.
# 
# Display the `.nunique()` method associated with `dfA_viz_c` to the screen to confirm it was created correctly.
# 
# #### 2g) - SOLUTION

# In[23]:


# Get the min and max values for x1 and x2 from the training data
x1_min = dfA['x1'].min()
x1_max = dfA['x1'].max()
x2_min = dfA['x2'].min()
x2_max = dfA['x2'].max()

# Create 9 evenly spaced values for x1
x1_values = np.linspace(x1_min, x1_max, 9)

# Create 101 evenly spaced values for x2
x2_values = np.linspace(x2_min, x2_max, 101)

# Create all combinations of x1 and x2 values
# This creates a grid where each x1 value is paired with each x2 value
x1_mesh, x2_mesh = np.meshgrid(x1_values, x2_values)

# Create the DataFrame
dfA_viz_c = pd.DataFrame({
    'x1': x1_mesh.flatten(),  # Convert 2D mesh to 1D array
    'x2': x2_mesh.flatten()   # Convert 2D mesh to 1D array
})

# Display the number of unique values in each column
print(dfA_viz_c.nunique())


# In[ ]:





# ### 2h)
# 
# You will make predictions for the AVERAGE OUTPUT (TREND) on the NEW `dfA_viz_c` data using the `fit_A_int` model. The UNCERTAINTY INTERVAL bounds are not required for this problem. Therefore, the predictive TREND can be added as a new column to a DataFrame. 
# 
# Create a COPY of the `dfA_viz_c` DataFrame named `dfA_viz_copy`. Create a NEW column named `pred` within `dfA_viz_copy` that is assigned the predictive TREND on the NEW `dfA_viz_c` data using the `fit_A_int` model.
# 
# Display the head of `dfA_viz_copy` object to the screen to confirm it is created correctly.
# 
# #### 2h) - SOLUTION

# In[24]:


# Create a copy of the visualization dataframe
dfA_viz_copy = dfA_viz_c.copy()

# Generate predictions for the average output (trend)
# Note that we only need the mean predictions, not the confidence or prediction intervals
dfA_viz_copy['pred'] = fit_A_int.predict(dfA_viz_c)

# Display the head of the dataframe with predictions
dfA_viz_copy.head()


# ### 2i)
# 
# You now have everything necessary to visualize the predictions of the `fit_A_int` model on the third NEW data, `dfA_viz_c`! You must visualize the predictive trend with respect to the input `x2` for each unique value of `x1`.
# 
# You must create the visualization as a LINE chart using Seaborn. The lines must be COLORED by the `x1` variable and you MUST use a DIVERGING color palette. You must set the appropriate arguments to ensure ALL lines are shown.
# 
# #### 2h) - SOLUTION

# In[25]:


# Create a figure with appropriate size
plt.figure(figsize=(12, 8))

# Create a line plot using Seaborn with a diverging color palette
# x2 on x-axis, predicted values on y-axis, colored by x1
sns.lineplot(data=dfA_viz_copy, x='x2', y='pred', hue='x1', 
             palette='coolwarm',  # Diverging color palette
             lw=2,                # Line width
             sort=True,           # Ensure lines are sorted by x2
             hue_norm=(-2.5, 2.5), # Normalize hue scale for better color distribution
             legend='full')       # Show all lines in the legend

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Improve plot aesthetics
plt.title('Effect of x2 on Predicted y for Different Values of x1', fontsize=16)
plt.xlabel('x2', fontsize=14)
plt.ylabel('Predicted y', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# Add a colorbar legend with a descriptive title
norm = plt.Normalize(-2.5, 2.5)
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('x1 value', fontsize=12)

# Adjust legend for clarity
plt.legend(title='x1 value', title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()


# ### 2j)
# 
# You have created multiple visualizations to study the predictions from a model that includes an INTERACTION between 2 continuous inputs.
# 
# How would you describe the INFLUENCE of `x1` on the RELATIOSHIP between the TREND and `x2` based on your figures?
# 
# #### 2j) - SOLUTION
# 
# What do you think?

# Based on our visualizations, I would describe the influence of x1 on the relationship between the trend and x2 as follows:
# 
# The value of x1 fundamentally changes the direction and magnitude of the relationship between x2 and the predicted output (y). Specifically:
# 
# 1. **Direction reversal**: x1 determines whether the relationship between x2 and y is positive or negative:
#    - When x1 is negative: x2 has a negative relationship with y (decreasing slope)
#    - When x1 is positive: x2 has a positive relationship with y (increasing slope)
#    - When x1 is near zero: the relationship between x2 and y is nearly flat
# 
# 2. **Magnitude amplification**: The absolute value of x1 controls the strength of the relationship:
#    - As |x1| increases: the relationship between x2 and y becomes stronger (steeper slopes)
#    - The further x1 is from zero (in either direction), the more pronounced the effect of x2 on y
# 
# 3. **Pivot point**: All trend lines appear to intersect near x2 = 0, suggesting that the interaction effect is minimal when x2 is close to zero, regardless of x1's value
# 
# This profound influence of x1 on the x2-y relationship explains why the interaction model was dramatically superior (R-squared of 92.8%) compared to simpler models. The relationship between x2 and y cannot be understood without considering the value of x1 - they fundamentally interact to determine the output.
# 

# ## Problem 03
# 
# This is another REGRESSION problem involving 2 inputs, `x1` and `x2`. The output is named `y`. However, this new application involves MIXED inputs! One input is CATEGORICAL (non-numeric) and one input is NUMERIC. You will gain experience working with the MIXED input setting, learning how to visualize the data, interpret the coefficients, and make predictions.
# 
# ### 3a)
# 
# Read in the `hw11_probB.csv` CSV file and assign it to the `dfB` object.
# 
# Apply the `.info()` method to display useful information associated with the columns in the data set.
# 
# #### 3a) - SOLUTION

# In[26]:


# Read in the dataset and store it in dfB
dfB = pd.read_csv('hw11_probB.csv')

# Display information about the dataframe
dfB.info()


# ### 3b)
# 
# As mentioned at the start of the assignment, you will NOT perform a detailed visual exploration of the data. However, let's practice using PLOTS that will specifically help you explore the potential influence of the CATEGORICAL input in this regression task.
# 
# Create a bar chart that shows the counts for the CATEGORICAL input. You must create the bar chart using Seaborn.
# 
# Are the categories roughly balanced?
# 
# #### 3b) - SOLUTION

# In[27]:


# Create a bar chart showing counts for the categorical input (x2)
plt.figure(figsize=(10, 6))
sns.countplot(data=dfB, x='x2')

# Customize the plot for better readability
plt.title('Distribution of Categorical Input (x2)', fontsize=14)
plt.xlabel('x2 Categories', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Optional: Rotate x-axis labels if there are many categories or if they are long
plt.xticks(rotation=45, ha='right')

# Add grid lines for better readability of counts
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()


# Looking at the bar chart, the categories are somewhat balanced but not perfectly so:
# 
# - Category C has the highest count, with approximately 60 observations
# - Categories D and B each have around 50 observations
# - Category A has the lowest count, with approximately 39 observations
# 
# This means we have a slight imbalance across the four categories (A, B, C, D), with category C being most represented and category A being least represented. The difference between the most common (C) and least common (A) category is about 21 observations.
# 
# While not perfectly balanced, this distribution is still reasonably good for regression analysis. None of the categories are severely underrepresented, and the sample sizes are sufficient for each category. This relatively balanced distribution helps ensure that our model coefficients will be reliable for all categories and reduces the risk of certain categories having disproportionate influence on the model.
# 

# ### 3c) 
# 
# Create a BOXPLOT to visualize the RELATIONSHIP between the CONTINUOUS output `y` and the CATEGORICAL input.
# 
# You must create the BOXPLOT using Seaborn.
# 
# #### 3c) - SOLUTION

# In[28]:


# Create a boxplot to visualize the relationship between the categorical input (x2) and output (y)
plt.figure(figsize=(12, 7))
sns.boxplot(data=dfB, x='x2', y='y')

# Customize the plot for better readability
plt.title('Relationship Between Categorical Input (x2) and Output (y)', fontsize=14)
plt.xlabel('x2 Categories', fontsize=12)
plt.ylabel('y', fontsize=12)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# Add grid lines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Optional: Add individual points for more detail
sns.stripplot(data=dfB, x='x2', y='y', color='black', size=4, alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()


# ### 3d)
# 
# Create a POINT PLOT to visualize the RELATIONSHIP between the CONTINUOUS output `y` and the CATEGORICAL input.
# 
# #### 3d) - SOLUTION

# In[29]:


# Create a point plot to visualize the relationship between the categorical input (x2) and output (y)
plt.figure(figsize=(12, 7))
sns.pointplot(data=dfB, x='x2', y='y', ci=95, capsize=0.2)

# Customize the plot for better readability
plt.title('Relationship Between Categorical Input (x2) and Output (y)', fontsize=14)
plt.xlabel('x2 Categories', fontsize=12)
plt.ylabel('Average y Value', fontsize=12)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# Add grid lines for better readability
plt.grid(linestyle='--', alpha=0.5)

# Show the plot
plt.tight_layout()
plt.show()


# ### 3e)
# 
# Let's now examine the RELATIONSHIP between the CONTINUOUS output `y` and the NUMERIC input.
# 
# Create a TREND PLOT between the output and the NUMERIC input.
# 
# #### 3e) - SOLUTION

# In[30]:


# Create a trend plot to visualize the relationship between the numeric input (x1) and output (y)
plt.figure(figsize=(12, 7))

# Create scatter plot with regression line (trend)
sns.regplot(data=dfB, x='x1', y='y', 
            scatter_kws={'alpha': 0.5, 'color': 'blue'}, 
            line_kws={'color': 'red'})

# Customize the plot for better readability
plt.title('Relationship Between Numeric Input (x1) and Output (y)', fontsize=14)
plt.xlabel('x1', fontsize=12)
plt.ylabel('y', fontsize=12)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# Add a vertical line at x=0 for reference
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

# Add grid lines for better readability
plt.grid(linestyle='--', alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()


# ### 3f)
# 
# You have visually explored the relationship between the output and EACH input. The previous plots did NOT include BOTH inputs. However, you already know strategies to visually explore if the CATEGORICAL input influences the RELATIONSHIP between the continuous output and numeric input.
# 
# Create a TREND PLOT between the CONTINUOUS output `y` and the NUMERIC input which ACCOUNTS for the influence of the CATEGORICAL input.
# 
# #### 3f) - SOLUTION

# In[31]:


# Create a trend plot that accounts for both numeric and categorical inputs
plt.figure(figsize=(14, 8))

# Create scatter plot with regression lines for each category
sns.lmplot(data=dfB, x='x1', y='y', hue='x2', 
           palette='viridis',  # Distinct color palette
           height=8, aspect=1.5,  # Control figure size
           scatter_kws={'alpha': 0.6, 's': 50},  # Adjust point transparency and size
           line_kws={'linewidth': 2},  # Adjust line thickness
           ci=95,  # 95% confidence interval
           legend=True)

# Customize the plot for better readability
plt.title('Relationship Between x1 and y by Category (x2)', fontsize=16)
plt.xlabel('x1 (Numeric Input)', fontsize=14)
plt.ylabel('y (Output)', fontsize=14)

# Add reference lines
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Add grid lines
plt.grid(linestyle='--', alpha=0.3)

# Adjust legend
plt.legend(title='x2 Category', title_fontsize=12, fontsize=10, loc='best')

# Show the plot
plt.tight_layout()
plt.show()


# ### 3g)
# 
# Now that you have some idea about the output to input relationships, let's quantify those effects by fitting a linear model. You will fit a linear model that uses ADDITIVE features for BOTH inputs. Thus, you do not need to fit models with a single input in this problem. You will use BOTH inputs.
# 
# Fit a linear model that assumes LINEAR ADDITIVE FEATURES associated with BOTH inputs. Assign the fitted model to the `fit_B` object.
# 
# Display the COEFFICIENT PLOT associated with `fit_B`.
# 
# #### 3g) - SOLUTION

# In[32]:


# Fit a linear model with both inputs as additive features
fit_B = smf.ols(formula='y ~ x1 + x2', data=dfB).fit()

# Display the coefficient plot
fig, ax = my_coefplot(fit_B, title='Coefficients for Model with x1 and x2 (Categorical)')
plt.show()


# In[ ]:





# ### 3h)
# 
# How many coefficients were estimated for the model that included ADDITIVE features for BOTH inputs in the previous problem?
# 
# What conclusions can you draw from the COEFFICIENT plot? Are your conclusions consistent with your visualization in 3f)?
# 
# #### 3h) - SOLUTION
# 
# What do you think?

# The model that included additive features for both inputs (x1 and x2) estimated 5 coefficients: one intercept, one for x1, and three dummy variables for categories B, C, and D of x2 (with A as the reference level).
# 
# From the coefficient plot, I can conclude that:
# 1. x1 has a significant positive effect on y (coefficient ≈ +1.3)
# 2. Category D has a significant positive effect compared to reference category A (coefficient ≈ +2.0)
# 3. Category C has a slight positive effect that's not significantly different from A
# 4. Category B has a significant negative effect compared to A (coefficient ≈ -2.0)
# 
# These conclusions are entirely consistent with the visualization in 3f, which showed parallel regression lines (confirming the additive effect of x1 across all categories) with vertical shifts between categories in the exact same order and magnitude as the coefficient plot: D highest, C/A in the middle, and B lowest.
# 

# ### 3i)
# 
# Let's use predictions to help confirm your statements in 3h). As with the previous problem, predictions require defining a NEW data set. The goal here is to examine the influence of the CATEGORICAL input on the RELATIONSHIP between the output `y` and the NUMERIC (continuous) input. The predictions will include multiple values of the CATEGORICAL input. The visualization will therefore only show the predictive TREND rather than the predictive UNCERTAINTY intervals. (CMPINF 2120 and CMPINF 2130 show how to create figures that include the influence of secondary inputs on predictive uncertainty intervals.)
# 
# As in the previous problem, you must begin by defining a NEW data set. This data set will treat the NUMERIC (continuous) input as the primary input with the most unique values. The CATEGORICAL input will serve as the secondary input. The NEW data set must include ALL unique values of the CATEGORICAL input.  
# 
# Create a NEW Pandas DataFrame, `dfB_viz`, that contains a two columns with the SAME names as the INPUTS in `dfB`. The NUMERIC (continuous) input must consist of 101 evenly spaced values between the TRAINING set MINIMUM and training set MAXIMUM NUMERIC input values. The CATEGORICAL input must consist of ALL unique values associated with the CATEGORICAL (non-numeric) input. Thus, the CATEGORICAL input is NOT constant! Remember that the training set is contained in the `dfB` DataFrame.
# 
# Display the `.nunique()` method associated with `dfB_viz` to the screen to confirm it was created correctly.
# 
# #### 3i) - SOLUTION

# In[33]:


# Get the min and max values for the numeric input (x1) from the training data
x1_min = dfB['x1'].min()
x1_max = dfB['x1'].max()

# Create 101 evenly spaced values for x1
x1_values = np.linspace(x1_min, x1_max, 101)

# Get all unique values of the categorical input (x2)
x2_unique = dfB['x2'].unique()

# Create all combinations of x1 and x2 values
# This creates a grid where each x1 value is paired with each x2 category
x1_mesh, x2_mesh = np.meshgrid(x1_values, x2_unique)

# Create the DataFrame
dfB_viz = pd.DataFrame({
    'x1': x1_mesh.flatten(),  # Convert 2D mesh to 1D array
    'x2': x2_mesh.flatten()   # Convert 2D mesh to 1D array
})

# Display the number of unique values in each column
print(dfB_viz.nunique())


# In[ ]:





# ### 3j)
# 
# You will make predictions for the AVERAGE OUTPUT (TREND) on the NEW `dfB_viz` data using the `fit_B` model. The UNCERTAINTY INTERVAL bounds are not required for this problem. Therefore, the predictive TREND can be added as a new column to a DataFrame. 
# 
# Create a COPY of the `dfB_viz` DataFrame named `dfB_viz_copy`. Create a NEW column named `pred` within `dfB_viz_copy` that is assigned the predictive TREND on the NEW `dfB_viz` data using the `fit_B` model.
# 
# Display the head of `dfB_viz_copy` object to the screen to confirm it is created correctly.
# 
# #### 3j) - SOLUTION

# In[35]:


# Create a copy of the visualization dataframe
dfB_viz_copy = dfB_viz.copy()

# Generate predictions for the average output (trend)
dfB_viz_copy['pred'] = fit_B.predict(dfB_viz)

# Display the head of the dataframe with predictions
dfB_viz_copy.head()


# ### 3k)
# 
# You now have everything necessary to visualize the predictions of the `fit_B` model on the NEW data, `dfB_viz`! You must visualize the predictive trend with respect to the NUMERIC (continuous) input for each unique value of the CATEGORICAL (non-numeric) input.
# 
# You must create the visualization as a LINE chart using Seaborn. The lines must be COLORED by the CATEGORICAL (non-numeric) input. You may use the default color palette. You must set the appropriate arguments to ensure ALL lines are shown.
# 
# #### 3k) - SOLUTION

# In[36]:


# Create a figure with appropriate size
plt.figure(figsize=(12, 8))

# Create a line plot using Seaborn
# x1 on x-axis, predicted values on y-axis, colored by x2 (categorical)
sns.lineplot(data=dfB_viz_copy, x='x1', y='pred', hue='x2', 
             palette='viridis',  # Optional: you can use default palette by removing this
             lw=2,               # Line width for better visibility
             sort=True,          # Ensure lines are sorted by x1
             legend='full')      # Show all categories in the legend

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Improve plot aesthetics
plt.title('Predicted y vs x1 for Different Categories of x2', fontsize=16)
plt.xlabel('x1 (Numeric Input)', fontsize=14)
plt.ylabel('Predicted y', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust legend for clarity
plt.legend(title='x2 Category', title_fontsize=12, loc='best')

# Show the plot
plt.tight_layout()
plt.show()


# ### 3l)
# 
# Were the "lines" visualized in 3k) consistent with your COEFFICIENT PLOT in 3g) and the colored TREND PLOT in 3g)?
# 
# Based on those visualizations, what conclusions can you draw from ADDING a CATEGORICAL input to a CONTINUOUS input?
# 
# #### 3l) - SOLUTION
# 
# What do you think?

# The lines visualized in 3k) are completely consistent with both the coefficient plot in 3g) and the colored trend plot in 3f). All three visualizations show the same pattern: parallel lines with identical positive slopes (matching the x1 coefficient of ~1.3), vertically shifted by amounts that match the categorical coefficients (D highest at ~+2 from reference, C slightly above reference, A as reference, and B lowest at ~-2 from reference).
# 
# Adding a categorical input to a model with a continuous input allows us to capture group-specific baseline differences while maintaining the same relationship between the continuous predictor and the outcome across all groups. In this model, the categorical variable (x2) shifts the predicted values up or down depending on the category, but doesn't change how the continuous variable (x1) relates to the outcome - it creates parallel trend lines with identical slopes but different intercepts for each category.
# 

# ## Problem 04
# 
# This is another REGRESSION problem involving 2 inputs, `x1` and `x2`. The output is named `y`. You will continue working with CATEGORICAL INPUTS. You will repeat many of the actions from Problem 03. However, this time you will not focus on ADDITIVE features. Instead, you will ultimately fit a model that INTERACTS a CATEGORICAL input with a CONTINUOUS input!
# 
# ### 4a)
# 
# Read in the `hw11_probC.csv` CSV file and assign it to the `dfC` object.
# 
# Apply the `.info()` method to display useful information associated with the columns in the data set.
# 
# #### 4a) - SOLUTION

# In[37]:


# Read in the dataset and store it in dfC
dfC = pd.read_csv('hw11_probC.csv')

# Display information about the dataframe
dfC.info()


# ### 4b)
# 
# Let's practice using PLOTS that will specifically help you explore the potential influence of the CATEGORICAL input in this regression task.
# 
# Create a bar chart that shows the counts for the CATEGORICAL input. You must create the bar chart using Seaborn.
# 
# Are the categories roughly balanced?
# 
# #### 4b) - SOLUTION

# In[39]:


# Create a bar chart showing counts for the categorical input (x2)
plt.figure(figsize=(10, 6))
sns.countplot(data=dfC, x='x2')

# Customize the plot for better readability
plt.title('Distribution of Categorical Input (x2)', fontsize=14)
plt.xlabel('x2 Categories', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Optional: Rotate x-axis labels if needed
plt.xticks(rotation=45, ha='right')

# Add grid lines for better readability of counts
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()


# The categories are not balanced. There's clear variation in counts across the four categories:
# - Category C has the highest count (~90-95 observations)
# - Category A has the second highest count (~80 observations)
# - Category B has the third highest count (~65-70 observations)
# - Category D has the lowest count (~60-65 observations)
# 
# The ratio between the most frequent (C) and least frequent (D) category is approximately 1.5:1, indicating moderate imbalance in the distribution of categorical values.
# 

# ### 4c) 
# 
# Create a BOXPLOT to visualize the RELATIONSHIP between the CONTINUOUS output `y` and the CATEGORICAL input.
# 
# You must create the BOXPLOT using Seaborn.
# 
# #### 4c) - SOLUTION

# In[40]:


# Create a boxplot to visualize the relationship between the categorical input (x2) and output (y)
plt.figure(figsize=(12, 7))
sns.boxplot(data=dfC, x='x2', y='y')

# Customize the plot for better readability
plt.title('Relationship Between Categorical Input (x2) and Output (y)', fontsize=14)
plt.xlabel('x2 Categories', fontsize=12)
plt.ylabel('y', fontsize=12)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# Add grid lines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Optional: Add individual points for more detail
sns.stripplot(data=dfC, x='x2', y='y', color='black', size=4, alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()


# ### 4d)
# 
# Create a POINT PLOT to visualize the RELATIONSHIP between the CONTINUOUS output `y` and the CATEGORICAL input.
# 
# #### 4d) - SOLUTION

# In[41]:


# Create a point plot to visualize the relationship between the categorical input (x2) and output (y)
plt.figure(figsize=(12, 7))
sns.pointplot(data=dfC, x='x2', y='y', ci=95, capsize=0.2)

# Customize the plot for better readability
plt.title('Relationship Between Categorical Input (x2) and Output (y)', fontsize=14)
plt.xlabel('x2 Categories', fontsize=12)
plt.ylabel('Average y Value', fontsize=12)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# Add grid lines for better readability
plt.grid(linestyle='--', alpha=0.5)

# Show the plot
plt.tight_layout()
plt.show()


# ### 4e)
# 
# Let's now examine the RELATIONSHIP between the CONTINUOUS output `y` and the NUMERIC input.
# 
# Create a TREND PLOT between the output and the NUMERIC input.
# 
# #### 4e) - SOLUTION

# In[42]:


# Create a trend plot to visualize the relationship between the numeric input (x1) and output (y)
plt.figure(figsize=(12, 7))

# Create scatter plot with regression line (trend)
sns.regplot(data=dfC, x='x1', y='y', 
            scatter_kws={'alpha': 0.5, 'color': 'blue'}, 
            line_kws={'color': 'red'})

# Customize the plot for better readability
plt.title('Relationship Between Numeric Input (x1) and Output (y)', fontsize=14)
plt.xlabel('x1', fontsize=12)
plt.ylabel('y', fontsize=12)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# Add a vertical line at x=0 for reference
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

# Add grid lines for better readability
plt.grid(linestyle='--', alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()


# ### 4f)
# 
# You have visually explored the relationship between the output and EACH input. The previous plots did NOT include BOTH inputs. However, you already know strategies to visually explore if the CATEGORICAL input influences the RELATIONSHIP between the continuous output and numeric input.
# 
# Create a TREND PLOT between the CONTINUOUS output `y` and the NUMERIC input which ACCOUNTS for the influence of the CATEGORICAL input.
# 
# #### 4f) - SOLUTION

# In[43]:


# Create a trend plot that accounts for both numeric and categorical inputs
plt.figure(figsize=(14, 8))

# Create scatter plot with regression lines for each category
sns.lmplot(data=dfC, x='x1', y='y', hue='x2', 
           palette='viridis',  # Distinct color palette
           height=8, aspect=1.5,  # Control figure size
           scatter_kws={'alpha': 0.6, 's': 50},  # Adjust point transparency and size
           line_kws={'linewidth': 2},  # Adjust line thickness
           ci=95,  # 95% confidence interval
           legend=True)

# Customize the plot for better readability
plt.title('Relationship Between x1 and y by Category (x2)', fontsize=16)
plt.xlabel('x1 (Numeric Input)', fontsize=14)
plt.ylabel('y (Output)', fontsize=14)

# Add reference lines
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Add grid lines
plt.grid(linestyle='--', alpha=0.3)

# Adjust legend
plt.legend(title='x2 Category', title_fontsize=12, fontsize=10, loc='best')

# Show the plot
plt.tight_layout()
plt.show()


# ### 4g)
# 
# Now that you have some idea about the output to input relationships, let's quantify those effects by fitting a linear model. You will fit a linear model that uses includes INTERACTIONS between BOTH inputs. Thus, you do not need to fit models with a single input in this problem. You will use BOTH inputs.
# 
# Fit a linear model that involves the LINEAR **main effects** AND the INTERACTION between BOTH inputs. Assign the fitted model to the `fit_C` object.
# 
# Display the COEFFICIENT PLOT associated with `fit_C`.
# 
# #### 4g) - SOLUTION

# In[45]:


# Fit linear model with main effects and interaction
fit_C = smf.ols('y ~ x1 * x2', data=dfC).fit()

# Create coefficient plot
plt.figure(figsize=(14, 8))
coef_df = pd.DataFrame({
    'coef': fit_C.params,
    'err': fit_C.bse,
    'pvalue': fit_C.pvalues
})

# Drop intercept for better visualization
plot_coef = coef_df.drop('Intercept')

# Create the coefficient plot
plt.errorbar(x=plot_coef.index, y=plot_coef['coef'], 
             yerr=1.96*plot_coef['err'], fmt='o', capsize=5, markersize=10)

# Add reference line at y=0
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# Add grid lines
plt.grid(linestyle='--', alpha=0.3)

# Customize the plot
plt.title('Coefficient Plot for Model with Main Effects and Interaction', fontsize=16)
plt.ylabel('Coefficient Value', fontsize=14)
plt.xticks(rotation=45, fontsize=12)

# Adjust layout
plt.tight_layout()
plt.show()


# In[ ]:





# ### 4h)
# 
# How many coefficients were estimated for the model that included **main effects** AND INTERACTIONS between for BOTH inputs in the previous problem?
# 
# What conclusions can you draw from the COEFFICIENT plot? Are your conclusions consistent with your visualization in 4f)?
# 
# #### 4h) - SOLUTION
# 
# What do you think?

# The model estimated 8 coefficients: an intercept plus 7 visible coefficients in the plot (3 for x2 categories, 1 for x1, and 3 for interactions).
# 
# The coefficient plot confirms the strong interaction effects observed in our visualization. The main effects (x2[B], x2[C], x2[D], x1) aren't statistically significant, while the interactions show that x1's effect on y depends entirely on x2 category, with a significant negative effect for category B (x1:x2[B] ≈ -2.3) and positive effect for category D (x1:x2[D] ≈ 2.1). This perfectly matches the opposing slopes we saw in the trend plot, where B showed a steep downward trend and D showed a steep upward trend, while other categories remained relatively flat.
# 

# ### 4i)
# 
# Let's use predictions to help confirm your statements in 4h). As in the previous problem, you must begin by defining a NEW data set. This data set will treat the NUMERIC (continuous) input as the primary input with the most unique values. The CATEGORICAL input will serve as the secondary input. The NEW data set must include ALL unique values of the CATEGORICAL input.  
# 
# Create a NEW Pandas DataFrame, `dfC_viz`, that contains a two columns with the SAME names as the INPUTS in `dfC`. The NUMERIC (continuous) input must consist of 101 evenly spaced values between the TRAINING set MINIMUM and training set MAXIMUM NUMERIC input values. The CATEGORICAL input must consist of ALL unique values associated with the CATEGORICAL (non-numeric) input. Thus, the CATEGORICAL input is NOT constant! Remember that the training set is contained in the `dfC` DataFrame.
# 
# Display the `.nunique()` method associated with `dfC_viz` to the screen to confirm it was created correctly.
# 
# #### 4i) - SOLUTION

# In[46]:


# Get min and max values of the numeric input x1
x1_min = dfC['x1'].min()
x1_max = dfC['x1'].max()

# Create 101 evenly spaced values for x1
x1_range = np.linspace(x1_min, x1_max, 101)

# Get unique categories of x2
x2_categories = dfC['x2'].unique()

# Create empty lists to store the data
x1_values = []
x2_values = []

# For each category, add all x1 values
for category in x2_categories:
    x1_values.extend(x1_range)
    x2_values.extend([category] * 101)

# Create the new DataFrame
dfC_viz = pd.DataFrame({
    'x1': x1_values,
    'x2': x2_values
})

# Confirm the structure
print(dfC_viz.nunique())


# In[ ]:





# ### 4j)
# 
# You will make predictions for the AVERAGE OUTPUT (TREND) on the NEW `dfC_viz` data using the `fit_C` model. The UNCERTAINTY INTERVAL bounds are not required for this problem. Therefore, the predictive TREND can be added as a new column to a DataFrame. 
# 
# Create a COPY of the `dfC_viz` DataFrame named `dfC_viz_copy`. Create a NEW column named `pred` within `dfC_viz_copy` that is assigned the predictive TREND on the NEW `dfC_viz` data using the `fit_C` model.
# 
# Display the head of `dfC_viz_copy` object to the screen to confirm it is created correctly.
# 
# #### 4j) - SOLUTION

# In[47]:


# Create a copy of the visualization dataframe
dfC_viz_copy = dfC_viz.copy()

# Generate predictions using the fit_C model
dfC_viz_copy['pred'] = fit_C.predict(dfC_viz_copy)

# Display the head of the dataframe to confirm
dfC_viz_copy.head()


# ### 4k)
# 
# You now have everything necessary to visualize the predictions of the `fit_C` model on the NEW data, `dfC_viz`! You must visualize the predictive trend with respect to the NUMERIC (continuous) input for each unique value of the CATEGORICAL (non-numeric) input.
# 
# You must create the visualization as a LINE chart using Seaborn. The lines must be COLORED by the CATEGORICAL (non-numeric) input. You may use the default color palette. You must set the appropriate arguments to ensure ALL lines are shown.
# 
# #### 4k) - SOLUTION

# In[48]:


# Create a line plot showing predictive trends by category
plt.figure(figsize=(12, 8))

# Plot lines colored by the categorical input
sns.lineplot(data=dfC_viz_copy, x='x1', y='pred', hue='x2', 
             linewidth=2.5, 
             palette='viridis')  # Using viridis palette to match previous plot

# Customize the plot
plt.title('Predicted Relationship Between x1 and y by Category (x2)', fontsize=16)
plt.xlabel('x1 (Numeric Input)', fontsize=14)
plt.ylabel('Predicted y (Output)', fontsize=14)

# Add reference lines
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Add grid
plt.grid(linestyle='--', alpha=0.3)

# Adjust legend
plt.legend(title='x2 Category', title_fontsize=12, fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()


# ### 4l)
# 
# Were the "lines" visualized in 4k) consistent with your COEFFICIENT PLOT in 4g) and the colored TREND PLOT in 4g)?
# 
# Based on those visualizations, what conclusions can you draw from INTERACTING a CATEGORICAL input with a CONTINUOUS input?
# 
# #### 4l) - SOLUTION
# 
# What do you think?

# Yes, the lines visualized in 4k) are perfectly consistent with both the coefficient plot in 4g) and the colored trend plot from earlier. All three visualizations tell the same story:
# 
# 1. The coefficient plot showed significant interaction terms (x1:x2[B] strongly negative, x1:x2[D] strongly positive) while main effects were not significant.
# 
# 2. The colored trend plot showed opposing slopes for different categories with similar patterns.
# 
# 3. The prediction lines in 4k) clearly display these same relationships in a clean, model-based visualization.
# 
# Based on these visualizations, we can conclude that:
# 
# 1. Interactions between categorical and continuous inputs reveal how the effect of a continuous variable changes across different categories.
# 
# 2. Without modeling interactions, we would have missed the opposing effects completely - the overall flat relationship observed when combining all data masked the true underlying patterns.
# 
# 3. Including interactions allows the model to estimate different slopes for each category, capturing significant effects that would otherwise cancel out in a main-effects-only model.
# 
# 4. Interaction models provide a more nuanced understanding of the data, showing that the relationship between x1 and y is not universal but depends entirely on which category of x2 we're examining.
# 

# ## Problem 05
# 
# Regression is an important modeling task, but it is NOT the only type of modeling application! Another very important predictive modeling application is **BINARY CLASSIFICATION** which has the goal of CLASSIFYING a CATEGORICAL OUTPUT into one of two CATEGORIES. However, you will often find that the BINARY OUTCOME (output) is **encoded** as an integer where a value of 1 corresponds to the EVENT of interest and a value of 0 denotes the NON-EVENT (the other category). It is therefore paramount to use the appropriate modeling functions that will correctly treat the BINARY OUTCOME as a CATEGORICAL variable even though the output is denoted as an integer!
# 
# This problem introduces you to these issues and gives you practice working with the modeling functions. The data for the problem involves a single input named `x` and the BINARY OUTCOME you wish to classify is named `y`.
# 
# ### 5a)
# 
# Read in the `hw11_probD.csv` CSV file and assign it to the `dfD` object.
# 
# Apply the `.info()` method to display useful information associated with the columns in the data set.
# 
# What is the data type of the BINARY OUTCOME?
# 
# #### 5a) - SOLUTION

# In[49]:


# Read in the CSV file
dfD = pd.read_csv('hw11_probD.csv')

# Display information about the dataframe
dfD.info()


# ### 5b)
# 
# Display the number of unique values associated with the BINARY OUTCOME. Is it in fact BINARY?
# 
# #### 5b) - SOLUTION

# In[51]:


# Display the unique values
print("Unique values in y:", dfD['y'].unique())

# Display the number of unique values for y
print("Number of unique values in y:")
print(dfD['y'].nunique())


# Yes, it is indeed binary. The output shows there are exactly 2 unique values (0 and 1) in the y variable, confirming it's a proper binary outcome suitable for binary classification.

# ### 5c)
# 
# As mentioned at the start of the assignment, you will NOT perform a detailed visual exploration of the data. However, let's practice using PLOTS that will specifically help you explore the potential influence of the input on the BINARY OUTCOME.
# 
# Let's start with the BINARY OUTCOME though. Create a bar chart that shows the counts for the BINARY OUTCOME. You must create the bar chart using Seaborn.
# 
# Are the categories roughly balanced?
# 
# #### 5c) - SOLUTION

# In[52]:


# Create a bar chart for the binary outcome counts
plt.figure(figsize=(10, 6))
sns.countplot(x='y', data=dfD)

# Customize the plot
plt.title('Distribution of Binary Outcome (y)', fontsize=14)
plt.xlabel('y (Binary Outcome)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks([0, 1], ['0 (Non-Event)', '1 (Event)'])
plt.grid(axis='y', alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()


# Yes, the categories are very well balanced. The bar chart shows nearly identical counts for both categories:
# 
# - Category 0 (Non-Event): approximately 77 observations
# - Category 1 (Event): approximately 78 observations
# 
# This balanced distribution is ideal for classification problems as it avoids class imbalance issues that could bias the model toward the majority class. Having roughly equal representation of both classes will make it easier to evaluate the model's performance accurately.
# 

# ### 5d)
# 
# Visualize the RELATIONSHIP between the BINARY OUTCOME and the CONTINUOUS input using a TREND PLOT. You **must** use the DEFAULT arguments for the TREND PLOT and therefore you will create the INCORRECT TREND!!!
# 
# #### 5d) - SOLUTION

# In[53]:


# Create an incorrect trend plot using default arguments
plt.figure(figsize=(12, 7))

# Using regplot with default arguments (which will treat y as continuous)
sns.regplot(x='x', y='y', data=dfD)

# Customize the plot
plt.title('Incorrect Trend Plot: Relationship Between x and Binary y', fontsize=14)
plt.xlabel('x (Continuous Input)', fontsize=12)
plt.ylabel('y (Binary Outcome)', fontsize=12)
plt.grid(alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()


# ### 5e)
# 
# Visualize the RELATIONSHIP between the BINARY OUTCOME and the CONTINUOUS input again using a TREND PLOT. However, this time CHANGE the arguments to the TREND PLOT function so that a LOGISTIC REGRESSION trend line is used instead of the DEFAULT LINEAR REGRESSION trend line. You are therefore creating the CORRECT TREND PLOT for the BINARY OUTCOME.
# 
# #### 5e) - SOLUTION

# In[54]:


# Create a correct trend plot using logistic regression
plt.figure(figsize=(12, 7))

# Using regplot with logistic=True for binary outcome
sns.regplot(x='x', y='y', data=dfD, 
            logistic=True,  # Use logistic regression instead of linear
            scatter_kws={'alpha': 0.5})  # Make points slightly transparent

# Customize the plot
plt.title('Correct Trend Plot: Logistic Regression for Binary Outcome', fontsize=14)
plt.xlabel('x (Continuous Input)', fontsize=12)
plt.ylabel('Probability of y=1 (Event)', fontsize=12)
plt.ylim(-0.05, 1.05)  # Set y-axis limits to appropriate probability range
plt.grid(alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()


# ### 5f)
# 
# Look closely at the figures in 5d) and 5e). How does the TREND PLOT in 5d) reveal that it is INCORRECT for this BINARY CLASSIFICATION application?
# 
# #### 5f) - SOLUTION
# 
# What do you think?

# Looking closely at both trend plots, the linear trend in 5d) reveals it's incorrect for binary classification in several critical ways:
# 
# 1. **Impossible predictions**: The linear trend predicts values below 0 and above 1, which are impossible for probabilities. Notice how the line extends below 0 for x values above 2, predicting nonsensical negative probabilities.
# 
# 2. **Constant rate of change**: The linear model assumes the effect of x on y is constant throughout the range, but binary outcomes typically show nonlinear relationships where the effect changes at different values of x.
# 
# 3. **Poor fit to data pattern**: The data shows a clear nonlinear pattern where the transition between classes is more rapid around x=0, but the linear model can't capture this S-shaped relationship.
# 
# 4. **Misleading confidence intervals**: The linear model's confidence intervals extend outside the [0,1] range, suggesting impossible outcomes and misrepresenting prediction uncertainty.
# 
# 5. **Incorrect modeling assumption**: Binary outcomes follow a Bernoulli distribution, not a normal distribution, making linear regression's underlying assumptions invalid for this data.
# 
# The logistic model in 5e) addresses all these issues by constraining predictions to valid probabilities and modeling the nonlinear relationship appropriately.
# 

# ### 5g)
# 
# Let's now fit a LOGISTIC REGRESSION for this BINARY CLASSIFICATION task. As discussed in the lecture recordings, LOGISTIC REGRESSION does NOT model the AVERAGE OUTPUT directly. Instead, the model is applied to the LOG-ODDS RATIO. 
# 
# You must use the CORRECT statsmodels function to fit a LOGISTIC REGRESSION model that LINEARLY relates the CONTINUOUS input to the LOG-ODDS RATIO.
# 
# Assign the fitted model to the `fit_D` object.
# 
# Print the `.summary()` method associated with the `fit_D` object to the screen.
# 
# #### 5g) - SOLUTION

# In[55]:


# Use the logit function which models the log-odds ratio
fit_D = smf.logit('y ~ x', data=dfD).fit()

# Print the model summary
print(fit_D.summary())


# In[ ]:





# ### 5h)
# 
# The printed summary table allows you to identify the CONTINUOUS input is considered statistically significant. However, let's visualize the coefficient summaries instead using the COEFFICIENT PLOT! This works because LOGISTIC REGRESSION is a **Generalized Linear Model (GLM)** and thus many of the concepts and interpretations we learned about with linear models apply to logistic regression!
# 
# Display the COEFFICIENT PLOT associated with `fit_D`. Is the CONTINUOUS input considered STATISTICALLY SIGNIFICANT using the conventional thresholds?
# 
# #### 5h) - SOLUTION

# In[56]:


# Create coefficient plot for the logistic regression model
plt.figure(figsize=(10, 6))

# Create DataFrame with coefficient information
coef_df = pd.DataFrame({
    'coef': fit_D.params,
    'err': fit_D.bse,
    'pvalue': fit_D.pvalues
})

# Create the coefficient plot
plt.errorbar(x=coef_df.index, y=coef_df['coef'], 
             yerr=1.96*coef_df['err'], fmt='o', capsize=5, markersize=10)

# Add a horizontal reference line at y=0
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# Add grid lines
plt.grid(linestyle='--', alpha=0.3)

# Customize the plot
plt.title('Coefficient Plot for Logistic Regression Model', fontsize=14)
plt.ylabel('Coefficient Value (Log-Odds Scale)', fontsize=12)
plt.xticks(fontsize=12)

# Adjust layout
plt.tight_layout()
plt.show()


# Yes, the continuous input (x) is clearly statistically significant at conventional thresholds. 
# 
# The coefficient plot shows that:
# 1. The coefficient for x is approximately -2.22
# 2. The 95% confidence interval (error bars) ranges from about -2.9 to -1.5
# 3. This confidence interval does NOT cross zero
# 
# Since the confidence interval for x does not include zero, we can conclude that x has a statistically significant negative effect on the log-odds of y=1 at the conventional p<0.05 threshold. This is consistent with the p-value of 0.000 we saw in the summary table earlier.
# 
# In contrast, the intercept's confidence interval clearly crosses zero, indicating it is not statistically significant.
# 

# ### 5i)
# 
# Let's conclude this problem by visualizing the PREDICTIONS from the LOGISTIC REGRESSION model, `fit_D`. As with the previous problems you must first create a NEW data set to support the visualization.
# 
# However, compared to the previous problems you will NOT base the BOUNDS of the NEW visualization data set based on the TRAINING set. Instead, you will use values well outside the training set. I do **NOT** recommend doing this for most applications. Predictive models are not truly intended for **extrapolation**. You will do this here to help your understanding with what's going on with the LOGISTIC REGRESSION model. Thus, the extrapolation is being done for teaching purposes.
# 
# Create a NEW Pandas DataFrame, `dfD_viz`, that contains a single column named `x`. This column must consist of 501 evenly spaced values between -7 and 7.
# 
# Display the `.nunique()` method associated with `dfD_viz` to the screen to confirm it was created correctly.
# 
# #### 5i) - SOLUTION

# In[57]:


# Create a new DataFrame for visualization with extended range
x_viz = np.linspace(-7, 7, 501)
dfD_viz = pd.DataFrame({'x': x_viz})

# Confirm the structure
print(dfD_viz.nunique())


# In[ ]:





# ### 5j)
# 
# You will make predictions on the NEW `dfD_viz` data using the `fit_D` model. You are NOT including UNCERTAINTY INTERVAL bounds with the predictions and thus the predictions can be added as a new column to a DataFrame.
# 
# Create a COPY of the `dfD_viz` DataFrame named `dfD_viz_copy`. Create a NEW column named `pred` within `dfD_viz_copy` that is assigned the prediction on the NEW `dfD_viz` data using the `fit_D` model.
# 
# Display the head of `dfD_viz_copy` object to the screen to confirm it is created correctly.
# 
# #### 5j) - SOLUTION

# In[58]:


# Create a copy of the visualization dataframe
dfD_viz_copy = dfD_viz.copy()

# Generate predictions using the fit_D model
dfD_viz_copy['pred'] = fit_D.predict(dfD_viz_copy)

# Display the head of the dataframe to confirm
print(dfD_viz_copy.head())


# ### 5k)
# 
# Create a LINE chart using Seaborn to visualize the PREDICTIONS from `fit_D` on the NEW `dfD_viz` data with respect to the CONTINUOUS input.
# 
# #### 5k) - SOLUTION

# In[59]:


# Create a line plot showing the predicted probabilities
plt.figure(figsize=(12, 7))

# Plot the predicted probabilities
sns.lineplot(data=dfD_viz_copy, x='x', y='pred', linewidth=3, color='blue')

# Customize the plot
plt.title('Logistic Regression Predicted Probabilities', fontsize=16)
plt.xlabel('x (Continuous Input)', fontsize=14)
plt.ylabel('Predicted Probability of y=1 (Event)', fontsize=14)

# Add reference lines
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Boundary (p=0.5)')
plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
plt.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Add grid
plt.grid(linestyle='--', alpha=0.3)

# Set y-axis limits to show full probability range
plt.ylim(-0.05, 1.05)

# Add legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()


# ### 5l)
# 
# Are the PREDICTIONS from `fit_D` consistent with the COEFFICIENT PLOT displayed in 5h)?
# 
# #### 5l) - SOLUTION
# 
# What do you think?

# Yes, the predictions from fit_D are completely consistent with the coefficient plot in 5h):
# 
# 1. The coefficient plot showed a strong negative coefficient for x (approximately -2.22), indicating that as x increases, the log-odds of y=1 significantly decreases.
# 
# 2. This negative relationship is clearly reflected in the prediction plot, where higher x values correspond to lower probabilities of y=1, and the relationship follows the expected S-shaped curve.
# 
# 3. The intercept in the coefficient plot was close to zero (around 0.19) with a confidence interval crossing zero, suggesting the log-odds at x=0 is not significantly different from zero.
# 
# 4. This is consistent with the prediction plot where the curve crosses the 0.5 probability threshold (log-odds = 0) slightly to the left of x=0, confirming the small positive intercept value.
# 
# 5. The steepness of the S-curve reflects the magnitude of the x coefficient (-2.22), showing that a relatively small change in x produces a substantial change in predicted probability near the middle of the curve.
# 
# The prediction visualization perfectly translates the log-odds coefficients into the more intuitive probability scale, confirming that our model correctly implements the mathematical relationship we see in the coefficient plot.
# 

# ### 5m)
# 
# Why are the PREDICTIONS from `fit_D` **not** a STRAIGHT LINE even though you used a formula that involved a LINEAR RELATIONSHIP with the CONTINUOUS input? 
# 
# #### 5m) - SOLUTION
# 
# What do you think?

# The predictions aren't a straight line because logistic regression involves a two-step transformation process:
# 
# 1. The model creates a linear relationship in the log-odds space:
#    - Log-odds = β₀ + β₁x (this is linear)
#    - In our case: Log-odds = 0.19 - 2.22x
# 
# 2. This linear log-odds value is then transformed to probability space using the logistic (sigmoid) function:
#    - Probability = 1/(1 + e^(-log-odds))
# 
# This second transformation is nonlinear and produces the S-shaped curve we see in the plot. The sigmoid function compresses any input from negative infinity to positive infinity into the bounded range [0,1], creating the characteristic curve.
# 
# This approach allows logistic regression to:
# - Maintain a straightforward linear relationship in log-odds space (easy interpretation of coefficients)
# - Produce valid probability values between 0 and 1 (impossible with a straight line)
# - Model the nonlinear relationship between inputs and probability of binary outcomes
# 
# That's why we see a curved line despite using a linear formula - we're viewing the results in probability space after the nonlinear transformation.
# 

# In[ ]:




