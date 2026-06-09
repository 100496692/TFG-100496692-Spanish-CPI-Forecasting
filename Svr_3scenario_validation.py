"""
================================================================================
SECTION 5.4 - ML: SVR
3 Scenarios:
  1. Train 2003-2017 → Test 2018
  2. Train 2003-2019 → Test 2020-2021
  3. Train 2003-2021 → Test 2022-2024
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

df = pd.read_pickle('spanish_cpi_dataset_clean.pkl')

print("="*100)
print("SVR - 3 SCENARIO VALIDATION")
print("="*100)

#==============================================================================
# FEATURE ENGINEERING
#==============================================================================

all_exog = ['Euribor_12M', 'Food_Consumption', 'Tourism_Overnight',
            'Mortgages', 'IPI', 'EUR_USD']
cpi_lags = ['CPI_lag1', 'CPI_lag2', 'CPI_lag3', 'CPI_lag12']

df_feat = df.copy()
for lag in [1, 2, 3, 12]:
    df_feat[f'CPI_lag{lag}'] = df_feat['CPI_Variation'].shift(lag)
df_feat['Month'] = df_feat.index.month
df_feat = df_feat.dropna()

features_A = cpi_lags + ['Food_Consumption', 'Month']
features_B = cpi_lags + all_exog + ['Month']
target     = 'CPI_Variation'

#==============================================================================
# SCENARIOS
#==============================================================================

scenarios = [
    {'name': 'Pre-COVID Normal (2018)',               'label': '2018',
     'train_start': '2003-01-01', 'train_end': '2017-12-31',
     'test_start':  '2018-01-01', 'test_end':  '2018-12-31'},
    {'name': 'COVID Shock (2020-2021)',                'label': '2020-2021',
     'train_start': '2003-01-01', 'train_end': '2019-12-31',
     'test_start':  '2020-01-01', 'test_end':  '2021-12-31'},
    {'name': 'Post-Pandemic & Inflation (2022-2024)', 'label': '2022-2024',
     'train_start': '2003-01-01', 'train_end': '2021-12-31',
     'test_start':  '2022-01-01', 'test_end':  '2024-12-31'}
]

#==============================================================================
# HYPERPARAMETER TUNING
#==============================================================================

print("\n[1/3] Hyperparameter Tuning...")

train_tune  = df_feat[df_feat.index <= '2017-12-31']
sx_tune     = StandardScaler()
sy_tune     = StandardScaler()
X_tune_sc   = sx_tune.fit_transform(train_tune[features_B])
y_tune_sc   = sy_tune.fit_transform(train_tune[target].values.reshape(-1,1)).ravel()

param_grid = {
    'C':       [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.05, 0.1, 0.2],
    'kernel':  ['rbf', 'linear'],
    'gamma':   ['scale', 'auto']
}
tscv = TimeSeriesSplit(n_splits=5)
gs   = GridSearchCV(SVR(), param_grid, cv=tscv,
                    scoring='neg_mean_absolute_error', n_jobs=-1)
gs.fit(X_tune_sc, y_tune_sc)
best_params = gs.best_params_
print(f"  ✓ Best params: {best_params}")

#==============================================================================
# MULTI-SCENARIO EVALUATION
#==============================================================================

print("\n[2/3] Multi-Scenario Evaluation...")

all_results    = []
scenario_preds = {}

for sc in scenarios:
    print(f"\n  Scenario: {sc['name']}")

    train = df_feat[(df_feat.index >= sc['train_start']) & (df_feat.index <= sc['train_end'])]
    test  = df_feat[(df_feat.index >= sc['test_start'])  & (df_feat.index <= sc['test_end'])]

    y_train = train[target]
    y_test  = test[target]

    scenario_preds[sc['label']] = {'actual': y_test, 'dates': y_test.index}

    for version, features in [('A_Food', features_A), ('B_AllExog', features_B)]:
        sx = StandardScaler()
        sy = StandardScaler()
        X_train_sc = sx.fit_transform(train[features])
        y_train_sc = sy.fit_transform(y_train.values.reshape(-1,1)).ravel()
        X_test_sc  = sx.transform(test[features])

        svr = SVR(**best_params)
        svr.fit(X_train_sc, y_train_sc)
        pred = sy.inverse_transform(svr.predict(X_test_sc).reshape(-1,1)).ravel()

        mae  = np.mean(np.abs(y_test.values - pred))
        rmse = np.sqrt(np.mean((y_test.values - pred)**2))

        print(f"    SVR_{version}: MAE={mae:.4f} | RMSE={rmse:.4f}")

        scenario_preds[sc['label']][f'SVR_{version}'] = pred
        all_results.append({'Scenario': sc['name'], 'Label': sc['label'],
                            'Model': f'SVR_{version}', 'MAE': mae, 'RMSE': rmse})

#==============================================================================
# PLOTS
#==============================================================================

print("\n[3/3] Generating plots...")

fig, axes = plt.subplots(3, 1, figsize=(16, 14))
for idx, sc in enumerate(scenarios):
    ax    = axes[idx]
    label = sc['label']
    preds = scenario_preds[label]
    dates = preds['dates'].to_numpy()
    actual = preds['actual']

    mae_a = [r['MAE'] for r in all_results if r['Label']==label and r['Model']=='SVR_A_Food'][0]
    mae_b = [r['MAE'] for r in all_results if r['Label']==label and r['Model']=='SVR_B_AllExog'][0]

    ax.plot(dates, actual.values,           color='black',   linewidth=2.5, marker='o', markersize=4, label='Actual', zorder=5)
    ax.plot(dates, preds['SVR_A_Food'],     color='#F77F00', linewidth=2,   linestyle='--', label=f'SVR+Food (MAE:{mae_a:.4f})')
    ax.plot(dates, preds['SVR_B_AllExog'],  color='#9D4EDD', linewidth=2,   linestyle='--', label=f'SVR+AllExog (MAE:{mae_b:.4f})')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel('CPI Variation (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Scenario {idx+1}: {sc["name"]}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    if idx == 2:
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')

plt.suptitle('SVR — Multi-Scenario Validation', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figure_5_svr_multiscenario.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: figure_5_svr_multiscenario.png")
plt.close()

results_df = pd.DataFrame(all_results)
results_df.to_csv('svr_results_3scenarios.csv', index=False)

print("\n" + "="*100)
print("SVR SUMMARY")
print("="*100)
print(results_df[['Scenario','Model','MAE','RMSE']].to_string(index=False))
print("\n✅ SVR COMPLETED")
