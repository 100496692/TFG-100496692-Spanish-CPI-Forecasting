"""
================================================================================
SECTION 5.3 - CLASSICAL MODELS: SARIMA & SARIMAX
3 Scenarios:
  1. Train 2003-2017 → Test 2018        (Normal conditions)
  2. Train 2003-2019 → Test 2020-2021   (COVID shock)
  3. Train 2003-2021 → Test 2022-2024   (Post-pandemic + inflation)
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

df = pd.read_pickle('spanish_cpi_dataset_clean.pkl')

print("="*100)
print("SARIMA & SARIMAX - 3 SCENARIO VALIDATION")
print("="*100)

optimal_order          = (0, 0, 1)
optimal_seasonal_order = (0, 1, 1, 12)

#==============================================================================
# SCENARIOS
#==============================================================================

scenarios = [
    {
        'name':        'Pre-COVID Normal (2018)',
        'label':       '2018',
        'train_start': '2003-01-01',
        'train_end':   '2017-12-31',
        'test_start':  '2018-01-01',
        'test_end':    '2018-12-31'
    },
    {
        'name':        'COVID Shock (2020-2021)',
        'label':       '2020-2021',
        'train_start': '2003-01-01',
        'train_end':   '2019-12-31',
        'test_start':  '2020-01-01',
        'test_end':    '2021-12-31'
    },
    {
        'name':        'Post-Pandemic & Inflation (2022-2024)',
        'label':       '2022-2024',
        'train_start': '2003-01-01',
        'train_end':   '2021-12-31',
        'test_start':  '2022-01-01',
        'test_end':    '2024-12-31'
    }
]

#==============================================================================
# ESTIMATION & FORECAST
#==============================================================================

all_results        = []
scenario_preds     = {}

for sc in scenarios:
    print(f"\n{'='*100}")
    print(f"SCENARIO: {sc['name']}")
    print(f"  Train: {sc['train_start']} → {sc['train_end']}")
    print(f"  Test:  {sc['test_start']} → {sc['test_end']}")
    print('='*100)

    y_train = df[(df.index >= sc['train_start']) & (df.index <= sc['train_end'])]['CPI_Variation']
    y_test  = df[(df.index >= sc['test_start'])  & (df.index <= sc['test_end'])]['CPI_Variation']
    X_train = df[(df.index >= sc['train_start']) & (df.index <= sc['train_end'])][['Food_Consumption']]
    X_test  = df[(df.index >= sc['test_start'])  & (df.index <= sc['test_end'])][['Food_Consumption']]

    scenario_preds[sc['label']] = {'actual': y_test, 'dates': y_test.index}

    # --- SARIMA ---
    sarima_fit  = SARIMAX(y_train, order=optimal_order,
                          seasonal_order=optimal_seasonal_order,
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False, maxiter=200)

    sarima_pred = sarima_fit.get_forecast(steps=len(y_test)).predicted_mean
    sarima_mae  = np.mean(np.abs(y_test.values - sarima_pred.values))
    sarima_rmse = np.sqrt(np.mean((y_test.values - sarima_pred.values)**2))

    scenario_preds[sc['label']]['SARIMA'] = sarima_pred.values

    print(f"\n  SARIMA:  MAE={sarima_mae:.4f} | RMSE={sarima_rmse:.4f}")

    all_results.append({
        'Scenario': sc['name'], 'Label': sc['label'],
        'Model': 'SARIMA', 'MAE': sarima_mae, 'RMSE': sarima_rmse
    })

    # --- SARIMAX ---
    sarimax_fit  = SARIMAX(y_train, exog=X_train, order=optimal_order,
                           seasonal_order=optimal_seasonal_order,
                           enforce_stationarity=False,
                           enforce_invertibility=False).fit(disp=False, maxiter=200)

    sarimax_pred = sarimax_fit.get_forecast(steps=len(y_test), exog=X_test).predicted_mean
    sarimax_mae  = np.mean(np.abs(y_test.values - sarimax_pred.values))
    sarimax_rmse = np.sqrt(np.mean((y_test.values - sarimax_pred.values)**2))

    scenario_preds[sc['label']]['SARIMAX'] = sarimax_pred.values

    print(f"  SARIMAX: MAE={sarimax_mae:.4f} | RMSE={sarimax_rmse:.4f}")
    improvement = ((sarima_mae - sarimax_mae) / sarima_mae) * 100
    print(f"  SARIMAX vs SARIMA: {improvement:+.2f}%")

    all_results.append({
        'Scenario': sc['name'], 'Label': sc['label'],
        'Model': 'SARIMAX', 'MAE': sarimax_mae, 'RMSE': sarimax_rmse
    })

#==============================================================================
# FIGURE - 3 PANEL COMPARISON
#==============================================================================

fig, axes = plt.subplots(3, 1, figsize=(16, 14))

for idx, sc in enumerate(scenarios):
    ax    = axes[idx]
    label = sc['label']
    preds = scenario_preds[label]
    dates = preds['dates'].to_numpy()
    actual = preds['actual']

    sarima_mae  = [r['MAE'] for r in all_results if r['Label']==label and r['Model']=='SARIMA'][0]
    sarimax_mae = [r['MAE'] for r in all_results if r['Label']==label and r['Model']=='SARIMAX'][0]

    ax.plot(dates, actual.values,        color='black',   linewidth=2.5, marker='o', markersize=4, label='Actual', zorder=5)
    ax.plot(dates, preds['SARIMA'],      color='#E63946', linewidth=2,   linestyle='--', label=f'SARIMA (MAE: {sarima_mae:.4f})')
    ax.plot(dates, preds['SARIMAX'],     color='#2E86AB', linewidth=2,   linestyle='--', label=f'SARIMAX+Food (MAE: {sarimax_mae:.4f})')

    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel('CPI Variation (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Scenario {idx+1}: {sc["name"]}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    if idx == 2:
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')

plt.suptitle('SARIMA vs SARIMAX — Multi-Scenario Validation', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figure_5_classical_multiscenario.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: figure_5_classical_multiscenario.png")
plt.close()

#==============================================================================
# SUMMARY & SAVE
#==============================================================================

results_df = pd.DataFrame(all_results)
results_df.to_csv('classical_results_3scenarios.csv', index=False)

print("\n" + "="*100)
print("SUMMARY TABLE")
print("="*100)
pivot = results_df.pivot_table(index='Model', columns='Label', values='MAE')
print(pivot.round(4).to_string())

print("\n✅ CLASSICAL MODELS COMPLETED")
print("Files: classical_results_3scenarios.csv | figure_5_classical_multiscenario.png")
