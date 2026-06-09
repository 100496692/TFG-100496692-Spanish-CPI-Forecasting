"""
================================================================================
SECTION 5.2 - DESCRIPTIVE ANALYSIS
Part 1: Target Variable (CPI_Variation) Analysis
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Load data
df = pd.read_pickle('spanish_cpi_dataset_clean.pkl')

print("="*80)
print("DESCRIPTIVE ANALYSIS - CPI VARIATION (TARGET VARIABLE)")
print("="*80)

#==============================================================================
# FIGURE 1: TIME SERIES WITH KEY EVENTS
#==============================================================================

fig, ax = plt.subplots(figsize=(16, 6))

# Plot CPI variation
ax.plot(df.index.to_numpy(), df['CPI_Variation'].values, linewidth=1.5, color='#2E86AB', label='CPI Monthly Variation')
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

# Mark key economic events
events = {
    '2008-09-15': ('Financial Crisis', '#E63946'),
    '2011-07-01': ('Sovereign Debt Crisis', '#F77F00'),
    '2020-03-01': ('COVID-19 Pandemic', '#9D4EDD'),
    '2022-02-24': ('Ukraine War / Energy Crisis', '#06FFA5'),
    '2023-06-01': ('Inflation Peak', '#FF006E')
}

for date, (label, color) in events.items():
    ax.axvline(pd.Timestamp(date), color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(pd.Timestamp(date), ax.get_ylim()[1]*0.9, label, 
            rotation=90, verticalalignment='top', fontsize=9, color=color, fontweight='bold')

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('CPI Variation (%)', fontsize=12, fontweight='bold')
ax.set_title('Spanish CPI Monthly Variation (2003-2024)\nwith Key Economic Events', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure_5_1_cpi_timeseries.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure_5_1_cpi_timeseries.png")
plt.close()

#==============================================================================
# FIGURE 2: DISTRIBUTION ANALYSIS
#==============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram with KDE
axes[0].hist(df['CPI_Variation'], bins=30, edgecolor='black', alpha=0.7, color='#2E86AB', density=True)
df['CPI_Variation'].plot(kind='kde', ax=axes[0], color='#E63946', linewidth=2)
axes[0].axvline(df['CPI_Variation'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {df["CPI_Variation"].mean():.3f}%')
axes[0].axvline(df['CPI_Variation'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df["CPI_Variation"].median():.3f}%')
axes[0].set_xlabel('CPI Variation (%)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Density', fontsize=11, fontweight='bold')
axes[0].set_title('Distribution of CPI Variation', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Boxplot
bp = axes[1].boxplot(df['CPI_Variation'], vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('#2E86AB')
bp['boxes'][0].set_alpha(0.7)
axes[1].set_ylabel('CPI Variation (%)', fontsize=11, fontweight='bold')
axes[1].set_title('Boxplot of CPI Variation', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# Q-Q Plot
from scipy import stats
stats.probplot(df['CPI_Variation'].dropna(), dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure_5_2_cpi_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure_5_2_cpi_distribution.png")
plt.close()

#==============================================================================
# FIGURE 3: BOXPLOT BY YEAR
#==============================================================================

fig, ax = plt.subplots(figsize=(16, 6))

# Create year column
df_year = df.copy()
df_year['Year'] = df_year.index.year

# Boxplot by year
bp = df_year.boxplot(column='CPI_Variation', by='Year', ax=ax, patch_artist=True, return_type='dict')

# Color boxes by decade
for i, patch in enumerate(bp['CPI_Variation']['boxes']):
    year = df_year['Year'].unique()[i]
    if year < 2010:
        patch.set_facecolor('#A8DADC')
    elif year < 2020:
        patch.set_facecolor('#457B9D')
    else:
        patch.set_facecolor('#1D3557')

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('CPI Variation (%)', fontsize=12, fontweight='bold')
ax.set_title('CPI Monthly Variation by Year (2003-2024)', fontsize=14, fontweight='bold')
plt.suptitle('')  # Remove default title
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figure_5_3_cpi_by_year.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure_5_3_cpi_by_year.png")
plt.close()

#==============================================================================
# FIGURE 4: SEASONAL DECOMPOSITION
#==============================================================================

# Perform seasonal decomposition
decomposition = seasonal_decompose(df['CPI_Variation'], model='additive', period=12)

fig, axes = plt.subplots(4, 1, figsize=(16, 10))

# Original
axes[0].plot(df.index.to_numpy(), df['CPI_Variation'].values, linewidth=1.5, color='#2E86AB')
axes[0].set_ylabel('Original', fontsize=11, fontweight='bold')
axes[0].set_title('Seasonal Decomposition of CPI Variation', fontsize=14, fontweight='bold', pad=20)
axes[0].grid(True, alpha=0.3)

# Trend
axes[1].plot(decomposition.trend.index.to_numpy(), decomposition.trend.values, linewidth=2, color='#E63946')
axes[1].set_ylabel('Trend', fontsize=11, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Seasonal
axes[2].plot(decomposition.seasonal.index.to_numpy(), decomposition.seasonal.values, linewidth=1.5, color='#06FFA5')
axes[2].set_ylabel('Seasonal', fontsize=11, fontweight='bold')
axes[2].grid(True, alpha=0.3)

# Residual
axes[3].plot(decomposition.resid.index.to_numpy(), decomposition.resid.values, linewidth=1, color='#9D4EDD', alpha=0.7)
axes[3].set_ylabel('Residual', fontsize=11, fontweight='bold')
axes[3].set_xlabel('Date', fontsize=11, fontweight='bold')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure_5_4_seasonal_decomposition.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure_5_4_seasonal_decomposition.png")
plt.close()

#==============================================================================
# FIGURE 5: ACF AND PACF
#==============================================================================

fig, axes = plt.subplots(2, 1, figsize=(16, 8))

# ACF
plot_acf(df['CPI_Variation'].dropna(), lags=40, ax=axes[0], color='#2E86AB', alpha=0.7)
axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Lag', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# PACF
plot_pacf(df['CPI_Variation'].dropna(), lags=40, ax=axes[1], color='#E63946', alpha=0.7)
axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Lag', fontsize=11, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure_5_5_acf_pacf.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure_5_5_acf_pacf.png")
plt.close()

#==============================================================================
# SUMMARY STATISTICS
#==============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS - CPI VARIATION")
print("="*80)

stats_summary = pd.DataFrame({
    'Statistic': ['Mean', 'Median', 'Std. Dev.', 'Min', 'Max', 'Skewness', 'Kurtosis', 'Q1', 'Q3', 'IQR'],
    'Value': [
        df['CPI_Variation'].mean(),
        df['CPI_Variation'].median(),
        df['CPI_Variation'].std(),
        df['CPI_Variation'].min(),
        df['CPI_Variation'].max(),
        df['CPI_Variation'].skew(),
        df['CPI_Variation'].kurtosis(),
        df['CPI_Variation'].quantile(0.25),
        df['CPI_Variation'].quantile(0.75),
        df['CPI_Variation'].quantile(0.75) - df['CPI_Variation'].quantile(0.25)
    ]
})

print(stats_summary.to_string(index=False))

print("\n" + "="*80)
print("✅ CPI VARIATION ANALYSIS COMPLETED")
print("="*80)
print("\nGenerated figures:")
print("  1. figure_5_1_cpi_timeseries.png - Time series with economic events")
print("  2. figure_5_2_cpi_distribution.png - Distribution analysis")
print("  3. figure_5_3_cpi_by_year.png - Boxplot by year")
print("  4. figure_5_4_seasonal_decomposition.png - Seasonal decomposition")
print("  5. figure_5_5_acf_pacf.png - ACF and PACF plots")
