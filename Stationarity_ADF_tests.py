"""
================================================================================
SECTION 5.3 - CLASSICAL MODELS
Part 1: Stationarity Testing (Augmented Dickey-Fuller)
================================================================================
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_pickle('spanish_cpi_dataset_clean.pkl')

print("="*100)
print("STATIONARITY TESTING - AUGMENTED DICKEY-FULLER (ADF) TEST")
print("="*100)

def adf_test(series, name, alpha=0.05):
    """
    Perform Augmented Dickey-Fuller test
    
    H0: Series has a unit root (non-stationary)
    H1: Series is stationary
    """
    result = adfuller(series.dropna(), autolag='AIC')
    
    print(f"\n{'='*100}")
    print(f"ADF Test Results: {name}")
    print('='*100)
    print(f"ADF Statistic:        {result[0]:.6f}")
    print(f"p-value:              {result[1]:.6f}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"   {key:>4s}: {value:.6f}")
    
    print(f"\n{'─'*100}")
    if result[1] <= alpha:
        print(f"✓ REJECT H0 at {alpha*100:.0f}% significance → Series is STATIONARY")
        stationary = True
    else:
        print(f"✗ FAIL TO REJECT H0 at {alpha*100:.0f}% significance → Series is NON-STATIONARY")
        stationary = False
    print('─'*100)
    
    return {
        'variable': name,
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_1%': result[4]['1%'],
        'critical_5%': result[4]['5%'],
        'critical_10%': result[4]['10%'],
        'stationary': stationary
    }

#==============================================================================
# TEST ALL VARIABLES
#==============================================================================

results = []

# Target variable
results.append(adf_test(df['CPI_Variation'], 'CPI_Variation'))

# Exogenous variables
exog_vars = ['Euribor_12M', 'Food_Consumption', 'Tourism_Overnight', 
             'Mortgages', 'IPI', 'EUR_USD']

for var in exog_vars:
    results.append(adf_test(df[var], var))

#==============================================================================
# SUMMARY TABLE
#==============================================================================

print("\n" + "="*100)
print("SUMMARY TABLE - STATIONARITY TESTS")
print("="*100)

summary_df = pd.DataFrame(results)
summary_df['Decision'] = summary_df['stationary'].apply(lambda x: 'STATIONARY' if x else 'NON-STATIONARY')

print(summary_df[['variable', 'adf_statistic', 'p_value', 'critical_5%', 'Decision']].to_string(index=False))

#==============================================================================
# DIFFERENCING FOR NON-STATIONARY VARIABLES
#==============================================================================

print("\n" + "="*100)
print("FIRST DIFFERENCING FOR NON-STATIONARY VARIABLES")
print("="*100)

non_stationary = summary_df[~summary_df['stationary']]['variable'].tolist()

if non_stationary:
    print(f"\nNon-stationary variables detected: {', '.join(non_stationary)}")
    print("\nApplying first difference (∇₁)...")
    
    diff_results = []
    for var in non_stationary:
        if var in df.columns:
            diff_series = df[var].diff().dropna()
            diff_results.append(adf_test(diff_series, f"{var}_diff"))
    
    print("\n" + "="*100)
    print("SUMMARY - DIFFERENCED VARIABLES")
    print("="*100)
    diff_df = pd.DataFrame(diff_results)
    diff_df['Decision'] = diff_df['stationary'].apply(lambda x: 'STATIONARY' if x else 'NON-STATIONARY')
    print(diff_df[['variable', 'adf_statistic', 'p_value', 'critical_5%', 'Decision']].to_string(index=False))
else:
    print("\n✓ All variables are already stationary. No differencing needed.")

#==============================================================================
# SEASONAL DIFFERENCING (for CPI_Variation if needed)
#==============================================================================

print("\n" + "="*100)
print("SEASONAL DIFFERENCING (∇₁₂) - CPI_Variation")
print("="*100)

cpi_seasonal_diff = df['CPI_Variation'].diff(12).dropna()
seasonal_result = adf_test(cpi_seasonal_diff, 'CPI_Variation_seasonal_diff_12')

#==============================================================================
# RECOMMENDATIONS
#==============================================================================

print("\n" + "="*100)
print("MODELING RECOMMENDATIONS")
print("="*100)

cpi_stationary = summary_df[summary_df['variable'] == 'CPI_Variation']['stationary'].iloc[0]

if cpi_stationary:
    print("\n✓ CPI_Variation is already STATIONARY")
    print("  → Recommendation: d = 0 for SARIMA(p,d,q)")
    print("  → However, ACF shows seasonal spikes at lags 12, 24, 36")
    print("  → Recommendation: Apply seasonal differencing D = 1 for SARIMA(p,d,q)×(P,D,Q)₁₂")
else:
    print("\n✗ CPI_Variation is NON-STATIONARY")
    print("  → Recommendation: d = 1 for SARIMA(p,d,q)")

print("\nExogenous variables:")
for var in exog_vars:
    var_stationary = summary_df[summary_df['variable'] == var]['stationary'].iloc[0]
    if var_stationary:
        print(f"  ✓ {var}: STATIONARY → use as-is in SARIMAX")
    else:
        print(f"  ✗ {var}: NON-STATIONARY → use first difference in SARIMAX")

#==============================================================================
# SAVE RESULTS
#==============================================================================

summary_df.to_csv('adf_test_results.csv', index=False)
print("\n" + "="*100)
print("✅ STATIONARITY TESTING COMPLETED")
print("="*100)
print("\nResults saved to: adf_test_results.csv")
