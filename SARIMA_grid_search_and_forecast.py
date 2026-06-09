"""
================================================================================
SECTION 5.3 - CLASSICAL MODELS
Part 2: SARIMA Model Identification and Estimation
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import itertools
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_pickle('spanish_cpi_dataset_clean.pkl')

print("="*100)
print("SARIMA MODEL IDENTIFICATION AND ESTIMATION")
print("="*100)

#==============================================================================
# TRAIN-TEST SPLIT
#==============================================================================

# Train: 2003-2019 (204 months)
# Test:  2020-2024 (60 months)

train_end = '2019-12-31'
test_start = '2020-01-01'

train = df[df.index <= train_end]['CPI_Variation']
test = df[df.index > train_end]['CPI_Variation']

print(f"\nTrain period: {train.index[0].strftime('%Y-%m')} to {train.index[-1].strftime('%Y-%m')} ({len(train)} obs)")
print(f"Test period:  {test.index[0].strftime('%Y-%m')} to {test.index[-1].strftime('%Y-%m')} ({len(test)} obs)")

#==============================================================================
# GRID SEARCH FOR OPTIMAL SARIMA PARAMETERS
#==============================================================================

print("\n" + "="*100)
print("GRID SEARCH - SARIMA(p,d,q)×(P,D,Q)₁₂ SPECIFICATION")
print("="*100)

# Define parameter ranges based on ACF/PACF analysis
# From Figure 5.5: seasonal spikes at 12, 24, 36 → seasonal component needed
p_range = range(0, 3)  # AR order
d_range = [0, 1]       # Differencing (CPI already stationary but test both)
q_range = range(0, 3)  # MA order

P_range = [0, 1, 2]    # Seasonal AR
D_range = [1]          # Seasonal differencing (justified by ACF)
Q_range = [0, 1, 2]    # Seasonal MA
s = 12                 # Seasonal period (monthly data)

# Generate all combinations
pdq = list(itertools.product(p_range, d_range, q_range))
seasonal_pdq = list(itertools.product(P_range, D_range, Q_range, [s]))

print(f"\nTesting {len(pdq)} × {len(seasonal_pdq)} = {len(pdq) * len(seasonal_pdq)} model specifications...")
print("This may take several minutes...\n")

results = []
best_aic = np.inf
best_bic = np.inf
best_model_aic = None
best_model_bic = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            model = SARIMAX(train,
                           order=param,
                           seasonal_order=param_seasonal,
                           enforce_stationarity=False,
                           enforce_invertibility=False)
            
            fitted_model = model.fit(disp=False, maxiter=200)
            
            results.append({
                'order': param,
                'seasonal_order': param_seasonal,
                'AIC': fitted_model.aic,
                'BIC': fitted_model.bic,
                'Log-Likelihood': fitted_model.llf
            })
            
            if fitted_model.aic < best_aic:
                best_aic = fitted_model.aic
                best_model_aic = (param, param_seasonal)
            
            if fitted_model.bic < best_bic:
                best_bic = fitted_model.bic
                best_model_bic = (param, param_seasonal)
                
        except Exception as e:
            continue

print("="*100)
print("GRID SEARCH COMPLETED")
print("="*100)

print(f"\n✓ Best model by AIC: SARIMA{best_model_aic[0]}×{best_model_aic[1]} (AIC = {best_aic:.2f})")
print(f"✓ Best model by BIC: SARIMA{best_model_bic[0]}×{best_model_bic[1]} (BIC = {best_bic:.2f})")

# Save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('AIC').reset_index(drop=True)
results_df.to_csv('sarima_grid_search_results.csv', index=False)

# Display top 10 models
print("\n" + "="*100)
print("TOP 10 MODELS BY AIC")
print("="*100)
print(results_df.head(10).to_string(index=False))

#==============================================================================
# ESTIMATE BEST MODEL
#==============================================================================

print("\n" + "="*100)
print("ESTIMATING OPTIMAL SARIMA MODEL")
print("="*100)

# Use BIC as it's more parsimonious (penalizes complexity more)
final_order = best_model_bic[0]
final_seasonal_order = best_model_bic[1]

print(f"\nFinal specification: SARIMA{final_order}×{final_seasonal_order}")

final_model = SARIMAX(train,
                      order=final_order,
                      seasonal_order=final_seasonal_order,
                      enforce_stationarity=False,
                      enforce_invertibility=False)

final_fitted = final_model.fit(disp=False, maxiter=200)

print("\n" + final_fitted.summary().as_text())

#==============================================================================
# DIAGNOSTIC PLOTS
#==============================================================================

print("\n" + "="*100)
print("GENERATING DIAGNOSTIC PLOTS")
print("="*100)

fig = final_fitted.plot_diagnostics(figsize=(16, 10))
plt.tight_layout()
plt.savefig('figure_5_10_sarima_diagnostics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure_5_10_sarima_diagnostics.png")
plt.close()

#==============================================================================
# IN-SAMPLE FIT
#==============================================================================

print("\n" + "="*100)
print("IN-SAMPLE PERFORMANCE (2003-2019)")
print("="*100)

fitted_values = final_fitted.fittedvalues
residuals = train - fitted_values

mae_train = np.mean(np.abs(residuals))
rmse_train = np.sqrt(np.mean(residuals**2))
mape_train = np.mean(np.abs(residuals / train)) * 100

print(f"\nMAE:  {mae_train:.4f}")
print(f"RMSE: {rmse_train:.4f}")
print(f"MAPE: {mape_train:.2f}%")

#==============================================================================
# OUT-OF-SAMPLE FORECAST (2020-2024)
#==============================================================================

print("\n" + "="*100)
print("OUT-OF-SAMPLE FORECAST (2020-2024)")
print("="*100)

# Forecast
n_forecast = len(test)
forecast = final_fitted.get_forecast(steps=n_forecast)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int(alpha=0.05)

# Calculate errors
forecast_errors = test.values - forecast_mean.values
mae_test = np.mean(np.abs(forecast_errors))
rmse_test = np.sqrt(np.mean(forecast_errors**2))
mape_test = np.mean(np.abs(forecast_errors / test.values)) * 100

print(f"\nMAE:  {mae_test:.4f}")
print(f"RMSE: {rmse_test:.4f}")
print(f"MAPE: {mape_test:.2f}%")

#==============================================================================
# PLOT FORECAST
#==============================================================================

fig, ax = plt.subplots(figsize=(16, 6))

# Plot historical data
ax.plot(train.index.to_numpy(), train.values, label='Training Data (2003-2019)', color='blue', linewidth=1.5)
ax.plot(test.index.to_numpy(), test.values, label='Actual (2020-2024)', color='black', linewidth=2)

# Plot forecast
ax.plot(test.index.to_numpy(), forecast_mean.values, label=f'SARIMA{final_order}×{final_seasonal_order} Forecast', 
        color='red', linewidth=2, linestyle='--')

# Plot confidence interval
ax.fill_between(test.index.to_numpy(), 
                forecast_ci.iloc[:, 0].values, 
                forecast_ci.iloc[:, 1].values,
                alpha=0.2, color='red', label='95% Confidence Interval')

ax.axvline(pd.Timestamp('2019-12-31'), color='green', linestyle='--', linewidth=2, label='Train/Test Split')
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('CPI Variation (%)', fontsize=12, fontweight='bold')
ax.set_title(f'SARIMA{final_order}×{final_seasonal_order} Out-of-Sample Forecast', 
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure_5_11_sarima_forecast.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: figure_5_11_sarima_forecast.png")
plt.close()

#==============================================================================
# SAVE MODEL AND RESULTS
#==============================================================================

# Save forecast results
forecast_df = pd.DataFrame({
    'Date': test.index,
    'Actual': test.values,
    'Forecast': forecast_mean.values,
    'Lower_CI': forecast_ci.iloc[:, 0].values,
    'Upper_CI': forecast_ci.iloc[:, 1].values,
    'Error': forecast_errors
})

forecast_df.to_csv('sarima_forecast_results.csv', index=False)

# Save model summary
with open('sarima_model_summary.txt', 'w') as f:
    f.write("="*100 + "\n")
    f.write(f"OPTIMAL SARIMA MODEL: SARIMA{final_order}×{final_seasonal_order}\n")
    f.write("="*100 + "\n\n")
    f.write(final_fitted.summary().as_text())
    f.write("\n\n" + "="*100 + "\n")
    f.write("OUT-OF-SAMPLE PERFORMANCE (2020-2024)\n")
    f.write("="*100 + "\n")
    f.write(f"MAE:  {mae_test:.4f}\n")
    f.write(f"RMSE: {rmse_test:.4f}\n")
    f.write(f"MAPE: {mape_test:.2f}%\n")

print("\n" + "="*100)
print("✅ SARIMA MODELING COMPLETED")
print("="*100)
print("\nGenerated files:")
print("  1. sarima_grid_search_results.csv - All tested models")
print("  2. sarima_forecast_results.csv - Out-of-sample forecast")
print("  3. sarima_model_summary.txt - Model summary and metrics")
print("  4. figure_5_10_sarima_diagnostics.png - Diagnostic plots")
print("  5. figure_5_11_sarima_forecast.png - Forecast plot")
