"""
================================================================================
EXHAUSTIVE SARIMAX ANALYSIS - PRE-COVID VALIDATION
Train: 2003-2017 (180 months)
Test:  2018 only (12 months)
================================================================================
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_pickle('spanish_cpi_dataset_clean.pkl')

print("="*100)
print("EXHAUSTIVE SARIMAX VARIABLE SELECTION ANALYSIS")
print("="*100)

#==============================================================================
# TRAIN-TEST SPLIT (PRE-COVID)
#==============================================================================

train_end = '2017-12-31'
test_start = '2018-01-01'
test_end = '2018-12-31'

y_train = df[(df.index >= '2003-01-01') & (df.index <= train_end)]['CPI_Variation']
y_test = df[(df.index >= test_start) & (df.index <= test_end)]['CPI_Variation']

exog_vars = ['Euribor_12M', 'Food_Consumption', 'Tourism_Overnight', 
             'Mortgages', 'IPI', 'EUR_USD']

X_full = df[exog_vars]
X_train = X_full[(X_full.index >= '2003-01-01') & (X_full.index <= train_end)]
X_test = X_full[(X_full.index >= test_start) & (X_full.index <= test_end)]

print(f"\nTrain period: {y_train.index[0].strftime('%Y-%m')} to {y_train.index[-1].strftime('%Y-%m')} ({len(y_train)} months)")
print(f"Test period:  {y_test.index[0].strftime('%Y-%m')} to {y_test.index[-1].strftime('%Y-%m')} ({len(y_test)} months)")
print(f"\nTest year: 2018 only (avoiding COVID-19 disruption)")

#==============================================================================
# OPTIMAL SARIMA SPECIFICATION
#==============================================================================

# Use optimal SARIMA from previous analysis
optimal_order = (0, 0, 1)
optimal_seasonal_order = (0, 1, 1, 12)

print(f"\nUsing SARIMA specification: {optimal_order}×{optimal_seasonal_order}")

#==============================================================================
# 1. BASELINE: SARIMA WITHOUT EXOGENOUS
#==============================================================================

print("\n" + "="*100)
print("1. BASELINE - SARIMA (NO EXOGENOUS VARIABLES)")
print("="*100)

sarima_model = SARIMAX(y_train,
                       order=optimal_order,
                       seasonal_order=optimal_seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)

sarima_fitted = sarima_model.fit(disp=False, maxiter=200)

# Forecast 2018
sarima_forecast = sarima_fitted.get_forecast(steps=len(y_test))
sarima_pred = sarima_forecast.predicted_mean

# Metrics
sarima_errors = y_test.values - sarima_pred.values
sarima_mae = np.mean(np.abs(sarima_errors))
sarima_rmse = np.sqrt(np.mean(sarima_errors**2))

print(f"\nBaseline Performance (2018):")
print(f"  MAE:  {sarima_mae:.4f}")
print(f"  RMSE: {sarima_rmse:.4f}")

# Store baseline
baseline_mae = sarima_mae
baseline_rmse = sarima_rmse

results_summary = []

results_summary.append({
    'Model': 'SARIMA (baseline)',
    'Variables': 'None',
    'N_vars': 0,
    'MAE': sarima_mae,
    'RMSE': sarima_rmse,
    'MAE_vs_baseline': 0.0,
    'RMSE_vs_baseline': 0.0,
    'AIC': sarima_fitted.aic,
    'BIC': sarima_fitted.bic
})

#==============================================================================
# 2. INDIVIDUAL VARIABLES (6 models)
#==============================================================================

print("\n" + "="*100)
print("2. INDIVIDUAL EXOGENOUS VARIABLES (ONE AT A TIME)")
print("="*100)

for var in exog_vars:
    print(f"\nTesting: {var}")
    
    try:
        model = SARIMAX(y_train,
                       exog=X_train[[var]],
                       order=optimal_order,
                       seasonal_order=optimal_seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        
        fitted = model.fit(disp=False, maxiter=200)
        
        # Forecast
        forecast = fitted.get_forecast(steps=len(y_test), exog=X_test[[var]])
        pred = forecast.predicted_mean
        
        # Metrics
        errors = y_test.values - pred.values
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        
        # Improvement vs baseline
        mae_improvement = ((baseline_mae - mae) / baseline_mae) * 100
        rmse_improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
        
        # Coefficient significance
        pvalue = fitted.pvalues.get(var, np.nan)
        significant = pvalue < 0.05 if not np.isnan(pvalue) else False
        
        print(f"  MAE:  {mae:.4f} ({mae_improvement:+.2f}% vs baseline)")
        print(f"  RMSE: {rmse:.4f} ({rmse_improvement:+.2f}% vs baseline)")
        print(f"  p-value: {pvalue:.4f} {'✓ Significant' if significant else '✗ Not significant'}")
        
        results_summary.append({
            'Model': f'SARIMAX + {var}',
            'Variables': var,
            'N_vars': 1,
            'MAE': mae,
            'RMSE': rmse,
            'MAE_vs_baseline': mae_improvement,
            'RMSE_vs_baseline': rmse_improvement,
            'AIC': fitted.aic,
            'BIC': fitted.bic,
            'p_value': pvalue,
            'Significant': significant
        })
        
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:50]}")
        continue

#==============================================================================
# 3. PAIRS OF VARIABLES (15 models)
#==============================================================================

print("\n" + "="*100)
print("3. PAIRS OF EXOGENOUS VARIABLES (COMBINATIONS OF 2)")
print("="*100)

pairs = list(itertools.combinations(exog_vars, 2))
print(f"\nTesting {len(pairs)} combinations of 2 variables...")

for pair in pairs:
    var_list = list(pair)
    var_names = ' + '.join(var_list)
    
    try:
        model = SARIMAX(y_train,
                       exog=X_train[var_list],
                       order=optimal_order,
                       seasonal_order=optimal_seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        
        fitted = model.fit(disp=False, maxiter=200)
        
        # Forecast
        forecast = fitted.get_forecast(steps=len(y_test), exog=X_test[var_list])
        pred = forecast.predicted_mean
        
        # Metrics
        errors = y_test.values - pred.values
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        
        # Improvement vs baseline
        mae_improvement = ((baseline_mae - mae) / baseline_mae) * 100
        rmse_improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
        
        # Count significant variables
        n_significant = sum(fitted.pvalues.get(v, 1.0) < 0.05 for v in var_list)
        
        results_summary.append({
            'Model': f'SARIMAX ({var_names})',
            'Variables': var_names,
            'N_vars': 2,
            'MAE': mae,
            'RMSE': rmse,
            'MAE_vs_baseline': mae_improvement,
            'RMSE_vs_baseline': rmse_improvement,
            'AIC': fitted.aic,
            'BIC': fitted.bic,
            'N_significant': n_significant
        })
        
    except Exception as e:
        continue

print(f"  ✓ Completed testing {len(pairs)} pairs")

#==============================================================================
# 4. TRIPLETS OF VARIABLES (20 models)
#==============================================================================

print("\n" + "="*100)
print("4. TRIPLETS OF EXOGENOUS VARIABLES (COMBINATIONS OF 3)")
print("="*100)

triplets = list(itertools.combinations(exog_vars, 3))
print(f"\nTesting {len(triplets)} combinations of 3 variables...")

for triplet in triplets:
    var_list = list(triplet)
    var_names = ' + '.join([v[:4] for v in var_list])  # Abbreviated names
    
    try:
        model = SARIMAX(y_train,
                       exog=X_train[var_list],
                       order=optimal_order,
                       seasonal_order=optimal_seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        
        fitted = model.fit(disp=False, maxiter=200)
        
        # Forecast
        forecast = fitted.get_forecast(steps=len(y_test), exog=X_test[var_list])
        pred = forecast.predicted_mean
        
        # Metrics
        errors = y_test.values - pred.values
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        
        # Improvement vs baseline
        mae_improvement = ((baseline_mae - mae) / baseline_mae) * 100
        rmse_improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
        
        # Count significant variables
        n_significant = sum(fitted.pvalues.get(v, 1.0) < 0.05 for v in var_list)
        
        results_summary.append({
            'Model': f'SARIMAX ({var_names})',
            'Variables': ' + '.join(var_list),
            'N_vars': 3,
            'MAE': mae,
            'RMSE': rmse,
            'MAE_vs_baseline': mae_improvement,
            'RMSE_vs_baseline': rmse_improvement,
            'AIC': fitted.aic,
            'BIC': fitted.bic,
            'N_significant': n_significant
        })
        
    except Exception as e:
        continue

print(f"  ✓ Completed testing {len(triplets)} triplets")

#==============================================================================
# 5. ALL VARIABLES (1 model)
#==============================================================================

print("\n" + "="*100)
print("5. ALL EXOGENOUS VARIABLES (FULL MODEL)")
print("="*100)

try:
    model = SARIMAX(y_train,
                   exog=X_train[exog_vars],
                   order=optimal_order,
                   seasonal_order=optimal_seasonal_order,
                   enforce_stationarity=False,
                   enforce_invertibility=False)
    
    fitted = model.fit(disp=False, maxiter=200)
    
    # Forecast
    forecast = fitted.get_forecast(steps=len(y_test), exog=X_test[exog_vars])
    pred = forecast.predicted_mean
    
    # Metrics
    errors = y_test.values - pred.values
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    # Improvement vs baseline
    mae_improvement = ((baseline_mae - mae) / baseline_mae) * 100
    rmse_improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
    
    # Count significant variables
    n_significant = sum(fitted.pvalues.get(v, 1.0) < 0.05 for v in exog_vars)
    
    print(f"\nFull Model Performance (2018):")
    print(f"  MAE:  {mae:.4f} ({mae_improvement:+.2f}% vs baseline)")
    print(f"  RMSE: {rmse:.4f} ({rmse_improvement:+.2f}% vs baseline)")
    print(f"  Significant variables: {n_significant}/{len(exog_vars)}")
    
    results_summary.append({
        'Model': 'SARIMAX (all 6 vars)',
        'Variables': ' + '.join(exog_vars),
        'N_vars': 6,
        'MAE': mae,
        'RMSE': rmse,
        'MAE_vs_baseline': mae_improvement,
        'RMSE_vs_baseline': rmse_improvement,
        'AIC': fitted.aic,
        'BIC': fitted.bic,
        'N_significant': n_significant
    })
    
except Exception as e:
    print(f"  ✗ Full model failed: {str(e)[:100]}")

#==============================================================================
# SUMMARY AND RANKING
#==============================================================================

print("\n" + "="*100)
print("COMPREHENSIVE RESULTS SUMMARY")
print("="*100)

results_df = pd.DataFrame(results_summary)

# Sort by MAE (best first)
results_df_sorted = results_df.sort_values('MAE').reset_index(drop=True)

print("\n" + "="*100)
print("TOP 10 MODELS BY MAE (2018 FORECAST ACCURACY)")
print("="*100)
print(results_df_sorted[['Model', 'N_vars', 'MAE', 'RMSE', 'MAE_vs_baseline', 'BIC']].head(10).to_string(index=False))

print("\n" + "="*100)
print("BOTTOM 5 MODELS (WORST PERFORMERS)")
print("="*100)
print(results_df_sorted[['Model', 'N_vars', 'MAE', 'RMSE', 'MAE_vs_baseline', 'BIC']].tail(5).to_string(index=False))

# Best models by category
print("\n" + "="*100)
print("BEST MODEL BY CATEGORY")
print("="*100)

for n in range(7):
    category = results_df[results_df['N_vars'] == n]
    if len(category) > 0:
        best = category.sort_values('MAE').iloc[0]
        print(f"\nBest with {n} variable(s):")
        print(f"  Model: {best['Model']}")
        print(f"  MAE:  {best['MAE']:.4f} ({best['MAE_vs_baseline']:+.2f}% vs baseline)")
        print(f"  RMSE: {best['RMSE']:.4f}")
        print(f"  BIC:  {best['BIC']:.2f}")

# Variables that improve when added individually
print("\n" + "="*100)
print("INDIVIDUAL VARIABLE PERFORMANCE RANKING")
print("="*100)

individual_vars = results_df[results_df['N_vars'] == 1].sort_values('MAE')
print(individual_vars[['Variables', 'MAE', 'MAE_vs_baseline', 'Significant']].to_string(index=False))

#==============================================================================
# MONTH-BY-MONTH COMPARISON (BEST MODEL VS BASELINE)
#==============================================================================

print("\n" + "="*100)
print("MONTH-BY-MONTH COMPARISON: BEST MODEL vs BASELINE (2018)")
print("="*100)

best_model_row = results_df_sorted.iloc[0]
print(f"\nBest overall model: {best_model_row['Model']}")
print(f"MAE: {best_model_row['MAE']:.4f} (improvement: {best_model_row['MAE_vs_baseline']:+.2f}%)")

# Re-estimate best model to get month-by-month predictions
best_vars = best_model_row['Variables']

if best_vars != 'None':
    best_var_list = [v.strip() for v in best_vars.split('+')]
    
    best_model = SARIMAX(y_train,
                        exog=X_train[best_var_list],
                        order=optimal_order,
                        seasonal_order=optimal_seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
    
    best_fitted = best_model.fit(disp=False, maxiter=200)
    best_forecast = best_fitted.get_forecast(steps=len(y_test), exog=X_test[best_var_list])
    best_pred = best_forecast.predicted_mean.values
else:
    best_pred = sarima_pred.values

# Create comparison table
comparison = pd.DataFrame({
    'Month': y_test.index.strftime('%Y-%m'),
    'Actual': y_test.values,
    'SARIMA': sarima_pred.values,
    'Best_Model': best_pred,
    'Error_SARIMA': y_test.values - sarima_pred.values,
    'Error_Best': y_test.values - best_pred
})

comparison['Winner'] = comparison.apply(
    lambda row: 'SARIMA' if abs(row['Error_SARIMA']) < abs(row['Error_Best']) else 'Best_Model', 
    axis=1
)

print("\n" + comparison.to_string(index=False))

# Count wins
sarima_wins = (comparison['Winner'] == 'SARIMA').sum()
best_wins = (comparison['Winner'] == 'Best_Model').sum()

print(f"\nMonth-by-month wins:")
print(f"  SARIMA: {sarima_wins}/12 months")
print(f"  {best_model_row['Model']}: {best_wins}/12 months")

#==============================================================================
# SAVE RESULTS
#==============================================================================

results_df_sorted.to_csv('exhaustive_sarimax_analysis_results.csv', index=False)
comparison.to_csv('month_by_month_comparison_2018.csv', index=False)

print("\n" + "="*100)
print("✅ EXHAUSTIVE ANALYSIS COMPLETED")
print("="*100)
print("\nGenerated files:")
print("  1. exhaustive_sarimax_analysis_results.csv - All model comparisons")
print("  2. month_by_month_comparison_2018.csv - Monthly forecast comparison")

#==============================================================================
# FINAL RECOMMENDATION
#==============================================================================

print("\n" + "="*100)
print("FINAL RECOMMENDATION")
print("="*100)

best_overall = results_df_sorted.iloc[0]

if best_overall['MAE_vs_baseline'] > 2.0:  # Improvement > 2%
    print(f"\n✓ RECOMMENDATION: Use {best_overall['Model']}")
    print(f"  Improves MAE by {best_overall['MAE_vs_baseline']:.2f}% vs baseline")
    print(f"  MAE: {best_overall['MAE']:.4f}")
    
    if 'Variables' in best_overall and best_overall['Variables'] != 'None':
        print(f"  Variables to include: {best_overall['Variables']}")
else:
    print(f"\n✗ RECOMMENDATION: Stick with SARIMA baseline")
    print(f"  Best SARIMAX improves MAE by only {best_overall['MAE_vs_baseline']:.2f}%")
    print(f"  Improvement is negligible - not worth added complexity")
    print(f"  Baseline MAE: {baseline_mae:.4f}")
