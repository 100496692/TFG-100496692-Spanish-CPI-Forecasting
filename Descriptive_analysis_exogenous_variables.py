"""
================================================================================
SECTION 5.2 - DESCRIPTIVE ANALYSIS
Part 2: Exogenous Variables Analysis
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Load data
df = pd.read_pickle('spanish_cpi_dataset_clean.pkl')

print("="*80)
print("DESCRIPTIVE ANALYSIS - EXOGENOUS VARIABLES")
print("="*80)

#==============================================================================
# FIGURE 6: TIME SERIES OF ALL EXOGENOUS VARIABLES
#==============================================================================

fig, axes = plt.subplots(3, 2, figsize=(18, 12))
axes = axes.flatten()

exogenous_vars = ['Euribor_12M', 'Food_Consumption', 'Tourism_Overnight', 
                  'Mortgages', 'IPI', 'EUR_USD']

titles = [
    'Euribor 12M Rate (%)',
    'Food Consumption (tonnes)',
    'Tourism Overnight Stays',
    'Residential Mortgages',
    'Industrial Production Index (2021=100)',
    'EUR/USD Exchange Rate'
]

colors = ['#E63946', '#F77F00', '#06FFA5', '#9D4EDD', '#457B9D', '#2E86AB']

for idx, (var, title, color) in enumerate(zip(exogenous_vars, titles, colors)):
    axes[idx].plot(df.index.to_numpy(), df[var].values, linewidth=1.5, color=color)
    axes[idx].set_title(title, fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Date', fontsize=10)
    axes[idx].set_ylabel(title.split('(')[0].strip(), fontsize=10)
    axes[idx].grid(True, alpha=0.3)
    
    # Mark COVID period
    axes[idx].axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-12-31'), 
                      alpha=0.2, color='red', label='COVID-19')
    
    # Add mean line
    axes[idx].axhline(df[var].mean(), color='black', linestyle='--', 
                     linewidth=1, alpha=0.5, label=f'Mean: {df[var].mean():.2f}')
    
    if idx == 0:  # Only show legend on first plot
        axes[idx].legend(loc='upper right', fontsize=8)

plt.suptitle('Exogenous Variables: Time Series (2003-2024)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figure_5_6_exogenous_timeseries.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure_5_6_exogenous_timeseries.png")
plt.close()

#==============================================================================
# FIGURE 7: CORRELATION MATRIX
#==============================================================================

fig, ax = plt.subplots(figsize=(12, 10))

# Calculate correlation matrix
corr_matrix = df.corr()

# Create heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=ax)

ax.set_title('Correlation Matrix of All Variables', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('figure_5_7_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure_5_7_correlation_matrix.png")
plt.close()

#==============================================================================
# FIGURE 8: SCATTER PLOTS - CPI vs EXOGENOUS VARIABLES
#==============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (var, title, color) in enumerate(zip(exogenous_vars, titles, colors)):
    # Scatter plot
    axes[idx].scatter(df[var].values, df['CPI_Variation'].values, alpha=0.6, s=30, color=color, edgecolors='black', linewidth=0.5)
    
    # Add regression line
    z = np.polyfit(df[var].dropna().values, df['CPI_Variation'].loc[df[var].dropna().index].values, 1)
    p = np.poly1d(z)
    axes[idx].plot(df[var].values, p(df[var].values), "r--", linewidth=2, alpha=0.8, label='Trend')
    
    # Calculate correlation
    corr = df[['CPI_Variation', var]].corr().iloc[0, 1]
    
    axes[idx].set_xlabel(title, fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('CPI Variation (%)', fontsize=10, fontweight='bold')
    axes[idx].set_title(f'{title.split("(")[0].strip()}\nCorr: {corr:.3f}', 
                       fontsize=11, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend(fontsize=8)

plt.suptitle('CPI Variation vs Exogenous Variables', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figure_5_8_scatter_plots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure_5_8_scatter_plots.png")
plt.close()

#==============================================================================
# FIGURE 9: DISTRIBUTION OF EXOGENOUS VARIABLES
#==============================================================================

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (var, title, color) in enumerate(zip(exogenous_vars, titles, colors)):
    # Histogram with KDE
    axes[idx].hist(df[var], bins=30, edgecolor='black', alpha=0.7, 
                  color=color, density=True)
    
    # KDE
    df[var].plot(kind='kde', ax=axes[idx], color='red', linewidth=2)
    
    # Mean and median lines
    axes[idx].axvline(df[var].mean(), color='green', linestyle='--', 
                     linewidth=2, label=f'Mean: {df[var].mean():.2f}')
    axes[idx].axvline(df[var].median(), color='orange', linestyle='--', 
                     linewidth=2, label=f'Median: {df[var].median():.2f}')
    
    axes[idx].set_xlabel(title, fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('Density', fontsize=10, fontweight='bold')
    axes[idx].set_title(f'Distribution: {title.split("(")[0].strip()}', 
                       fontsize=11, fontweight='bold')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Distribution of Exogenous Variables', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figure_5_9_exogenous_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure_5_9_exogenous_distributions.png")
plt.close()

#==============================================================================
# SUMMARY STATISTICS TABLE
#==============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS - EXOGENOUS VARIABLES")
print("="*80)

summary_stats = df[exogenous_vars].describe().T
summary_stats['skew'] = df[exogenous_vars].skew()
summary_stats['kurtosis'] = df[exogenous_vars].kurtosis()

print(summary_stats.to_string())

# Correlation with CPI
print("\n" + "="*80)
print("CORRELATION WITH CPI VARIATION")
print("="*80)

correlations = pd.DataFrame({
    'Variable': exogenous_vars,
    'Correlation': [df[['CPI_Variation', var]].corr().iloc[0, 1] for var in exogenous_vars]
}).sort_values('Correlation', key=abs, ascending=False)

print(correlations.to_string(index=False))

print("\n" + "="*80)
print("✅ EXOGENOUS VARIABLES ANALYSIS COMPLETED")
print("="*80)
print("\nGenerated figures:")
print("  6. figure_5_6_exogenous_timeseries.png - Time series panel")
print("  7. figure_5_7_correlation_matrix.png - Correlation heatmap")
print("  8. figure_5_8_scatter_plots.png - Scatter plots vs CPI")
print("  9. figure_5_9_exogenous_distributions.png - Distributions")
