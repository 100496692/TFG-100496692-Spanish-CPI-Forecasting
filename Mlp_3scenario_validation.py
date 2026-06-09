"""
================================================================================
SECTION 5.4 - ML: MULTILAYER PERCEPTRON (MLP)
3 Scenarios:
  1. Train 2003-2017 → Test 2018
  2. Train 2003-2019 → Test 2020-2021
  3. Train 2003-2021 → Test 2022-2024
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
    print("✓ TensorFlow imported successfully")
except ImportError:
    print("⚠️ Install: pip install tensorflow --break-system-packages")
    exit()

df = pd.read_pickle('spanish_cpi_dataset_clean.pkl')

print("="*100)
print("MULTILAYER PERCEPTRON (MLP) - 3 SCENARIO VALIDATION")
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

print(f"\n  ✓ Features A ({len(features_A)}): {features_A}")
print(f"  ✓ Features B ({len(features_B)}): {features_B}")

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
# MLP ARCHITECTURE
#==============================================================================

def build_mlp(input_dim, units_1=128, units_2=64, units_3=32,
              dropout=0.2, learning_rate=0.001):
    """
    MLP with 3 hidden layers, BatchNormalization, and Dropout
    Architecture: Input → 128 → 64 → 32 → 1
    """
    model = Sequential([
        Dense(units_1, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(dropout),
        Dense(units_2, activation='relu'),
        BatchNormalization(),
        Dropout(dropout),
        Dense(units_3, activation='relu'),
        Dropout(dropout / 2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mae')
    return model

EPOCHS      = 200
BATCH_SIZE  = 16
PATIENCE    = 20
DROPOUT     = 0.2
LR          = 0.001

early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE,
                           restore_best_weights=True, verbose=0)

#==============================================================================
# MULTI-SCENARIO EVALUATION
#==============================================================================

print("\nMulti-Scenario Evaluation...")

all_results       = []
scenario_preds    = {}
training_histories = {}  # Store for loss curve plots

for sc in scenarios:
    print(f"\n  {'='*80}")
    print(f"  Scenario: {sc['name']}")

    train = df_feat[(df_feat.index >= sc['train_start']) &
                    (df_feat.index <= sc['train_end'])]
    test  = df_feat[(df_feat.index >= sc['test_start']) &
                    (df_feat.index <= sc['test_end'])]

    y_train_raw = train[target].values
    y_test_raw  = test[target].values

    scenario_preds[sc['label']] = {
        'actual': test[target], 'dates': test.index
    }
    training_histories[sc['label']] = {}

    for version, features in [('A_Food', features_A), ('B_AllExog', features_B)]:
        print(f"\n    Training MLP version {version}...")

        X_train_raw = train[features].values
        X_test_raw  = test[features].values

        # Scale
        sx = StandardScaler()
        sy = StandardScaler()
        X_train_sc = sx.fit_transform(X_train_raw)
        y_train_sc = sy.fit_transform(y_train_raw.reshape(-1,1)).ravel()
        X_test_sc  = sx.transform(X_test_raw)

        # Validation split (last 15% of training)
        val_size   = max(int(len(X_train_sc) * 0.15), 12)
        X_tr, X_val = X_train_sc[:-val_size], X_train_sc[-val_size:]
        y_tr, y_val = y_train_sc[:-val_size], y_train_sc[-val_size:]

        tf.random.set_seed(42)
        model = build_mlp(input_dim=len(features), dropout=DROPOUT,
                          learning_rate=LR)

        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0
        )

        epochs_run = len(history.history['loss'])
        print(f"    Trained for {epochs_run} epochs (early stopping)")

        # Store history for loss curve
        training_histories[sc['label']][version] = history.history

        # Predict
        pred_sc = model.predict(X_test_sc, verbose=0).ravel()
        pred    = sy.inverse_transform(pred_sc.reshape(-1,1)).ravel()

        mae  = np.mean(np.abs(y_test_raw - pred))
        rmse = np.sqrt(np.mean((y_test_raw - pred)**2))
        r2   = 1 - np.sum((y_test_raw - pred)**2) / \
                   np.sum((y_test_raw - np.mean(y_test_raw))**2)

        print(f"    MLP_{version}: MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")

        scenario_preds[sc['label']][f'MLP_{version}'] = pred
        all_results.append({
            'Scenario': sc['name'], 'Label': sc['label'],
            'Model':    f'MLP_{version}',
            'MAE': mae, 'RMSE': rmse, 'R2': r2,
            'Epochs': epochs_run
        })

        tf.keras.backend.clear_session()

#==============================================================================
# FIGURE 1: TRAINING vs VALIDATION LOSS CURVES (Version A only)
#==============================================================================

print("\nGenerating training loss curves...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, sc in enumerate(scenarios):
    ax      = axes[idx]
    label   = sc['label']
    history = training_histories[label]['A_Food']

    epochs  = range(1, len(history['loss']) + 1)

    ax.plot(epochs, history['loss'],     color='#2E86AB', linewidth=2,
            label='Training Loss')
    ax.plot(epochs, history['val_loss'], color='#E63946', linewidth=2,
            linestyle='--', label='Validation Loss')

    best_epoch = np.argmin(history['val_loss']) + 1
    ax.axvline(best_epoch, color='green', linestyle=':', linewidth=1.5,
               label=f'Best epoch: {best_epoch}')

    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('MAE Loss', fontsize=11, fontweight='bold')
    ax.set_title(f'Scenario {idx+1}: {sc["name"]}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('MLP Training vs Validation Loss (Version A)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('figure_5_mlp_loss_curves.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: figure_5_mlp_loss_curves.png")
plt.close()

#==============================================================================
# FIGURE 2: FORECAST PLOTS
#==============================================================================

fig, axes = plt.subplots(3, 1, figsize=(16, 14))

for idx, sc in enumerate(scenarios):
    ax     = axes[idx]
    label  = sc['label']
    preds  = scenario_preds[label]
    dates  = preds['dates'].to_numpy()
    actual = preds['actual']

    mae_a = [r['MAE'] for r in all_results
             if r['Label']==label and r['Model']=='MLP_A_Food'][0]
    mae_b = [r['MAE'] for r in all_results
             if r['Label']==label and r['Model']=='MLP_B_AllExog'][0]

    ax.plot(dates, actual.values,           color='black',   linewidth=2.5,
            marker='o', markersize=4, label='Actual', zorder=5)
    ax.plot(dates, preds['MLP_A_Food'],     color='#F77F00', linewidth=2,
            linestyle='--', label=f'MLP+Food (MAE:{mae_a:.4f})')
    ax.plot(dates, preds['MLP_B_AllExog'],  color='#9D4EDD', linewidth=2,
            linestyle='--', label=f'MLP+AllExog (MAE:{mae_b:.4f})')

    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel('CPI Variation (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Scenario {idx+1}: {sc["name"]}',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    if idx == 2:
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')

plt.suptitle('MLP — Multi-Scenario Out-of-Sample Forecasts',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figure_5_mlp_multiscenario.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: figure_5_mlp_multiscenario.png")
plt.close()

#==============================================================================
# SAVE RESULTS
#==============================================================================

results_df = pd.DataFrame(all_results)
results_df.to_csv('mlp_results_3scenarios.csv', index=False)

print("\n" + "="*100)
print("MLP RESULTS SUMMARY")
print("="*100)
print(results_df[['Scenario','Model','MAE','RMSE','R2','Epochs']].to_string(index=False))

print("\n✅ MLP COMPLETED")
print("Files: mlp_results_3scenarios.csv | figure_5_mlp_loss_curves.png | figure_5_mlp_multiscenario.png")
