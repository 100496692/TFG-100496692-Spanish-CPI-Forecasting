"""
================================================================================
SECTION 5.4 - ML: LSTM
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
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
    print("✓ TensorFlow imported successfully")
except ImportError:
    print("⚠️ Install: pip install tensorflow --break-system-packages")
    exit()

df = pd.read_pickle('spanish_cpi_dataset_clean.pkl')

print("="*100)
print("LSTM - 3 SCENARIO VALIDATION")
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

TIMESTEPS     = 12
UNITS         = 64
DROPOUT       = 0.2
LEARNING_RATE = 0.001
EPOCHS        = 100
BATCH_SIZE    = 16

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
# HELPERS
#==============================================================================

def build_lstm(input_shape):
    model = Sequential([
        LSTM(UNITS, return_sequences=True, input_shape=input_shape),
        Dropout(DROPOUT),
        LSTM(UNITS // 2, return_sequences=False),
        Dropout(DROPOUT),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mae')
    return model

def make_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(timesteps, len(X)):
        Xs.append(X[i-timesteps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

early_stop = EarlyStopping(monitor='val_loss', patience=15,
                           restore_best_weights=True, verbose=0)

#==============================================================================
# MULTI-SCENARIO EVALUATION
#==============================================================================

print("\nMulti-Scenario Evaluation...")

all_results    = []
scenario_preds = {}

for sc in scenarios:
    print(f"\n  {'='*80}")
    print(f"  Scenario: {sc['name']}")

    train = df_feat[(df_feat.index >= sc['train_start']) & (df_feat.index <= sc['train_end'])]
    test  = df_feat[(df_feat.index >= sc['test_start'])  & (df_feat.index <= sc['test_end'])]

    y_train_raw = train[target].values
    y_test_raw  = test[target].values

    scenario_preds[sc['label']] = {'actual': test[target], 'dates': test.index}

    for version, features in [('A_Food', features_A), ('B_AllExog', features_B)]:
        print(f"\n    Training LSTM_{version}...")

        X_train_raw = train[features].values
        X_test_raw  = test[features].values

        sx = StandardScaler()
        sy = StandardScaler()

        X_all_sc   = sx.fit_transform(np.vstack([X_train_raw, X_test_raw]))
        X_train_sc = X_all_sc[:len(X_train_raw)]
        X_test_sc  = X_all_sc[len(X_train_raw):]
        y_train_sc = sy.fit_transform(y_train_raw.reshape(-1,1)).ravel()

        X_seq_tr, y_seq_tr = make_sequences(X_train_sc, y_train_sc, TIMESTEPS)

        X_comb = np.vstack([X_train_sc[-TIMESTEPS:], X_test_sc])
        X_seq_te, _ = make_sequences(X_comb, np.zeros(len(X_comb)), TIMESTEPS)

        val_split = max(int(len(X_seq_tr) * 0.15), TIMESTEPS)

        tf.random.set_seed(42)
        model = build_lstm((TIMESTEPS, len(features)))

        model.fit(X_seq_tr[:-val_split], y_seq_tr[:-val_split],
                  validation_data=(X_seq_tr[-val_split:], y_seq_tr[-val_split:]),
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=[early_stop], verbose=0)

        pred_sc = model.predict(X_seq_te, verbose=0).ravel()
        pred    = sy.inverse_transform(pred_sc.reshape(-1,1)).ravel()[:len(y_test_raw)]

        mae  = np.mean(np.abs(y_test_raw - pred))
        rmse = np.sqrt(np.mean((y_test_raw - pred)**2))

        print(f"    LSTM_{version}: MAE={mae:.4f} | RMSE={rmse:.4f}")

        scenario_preds[sc['label']][f'LSTM_{version}'] = pred
        all_results.append({'Scenario': sc['name'], 'Label': sc['label'],
                            'Model': f'LSTM_{version}', 'MAE': mae, 'RMSE': rmse})

        tf.keras.backend.clear_session()

#==============================================================================
# PLOTS
#==============================================================================

fig, axes = plt.subplots(3, 1, figsize=(16, 14))
for idx, sc in enumerate(scenarios):
    ax    = axes[idx]
    label = sc['label']
    preds = scenario_preds[label]
    dates = preds['dates'].to_numpy()
    actual = preds['actual']

    mae_a = [r['MAE'] for r in all_results if r['Label']==label and r['Model']=='LSTM_A_Food'][0]
    mae_b = [r['MAE'] for r in all_results if r['Label']==label and r['Model']=='LSTM_B_AllExog'][0]

    ax.plot(dates, actual.values,            color='black',   linewidth=2.5, marker='o', markersize=4, label='Actual', zorder=5)
    ax.plot(dates, preds['LSTM_A_Food'],     color='#F77F00', linewidth=2,   linestyle='--', label=f'LSTM+Food (MAE:{mae_a:.4f})')
    ax.plot(dates, preds['LSTM_B_AllExog'],  color='#9D4EDD', linewidth=2,   linestyle='--', label=f'LSTM+AllExog (MAE:{mae_b:.4f})')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel('CPI Variation (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'Scenario {idx+1}: {sc["name"]}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    if idx == 2:
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')

plt.suptitle('LSTM — Multi-Scenario Validation', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figure_5_lstm_multiscenario.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: figure_5_lstm_multiscenario.png")
plt.close()

results_df = pd.DataFrame(all_results)
results_df.to_csv('lstm_results_3scenarios.csv', index=False)

print("\n" + "="*100)
print("LSTM SUMMARY")
print("="*100)
print(results_df[['Scenario','Model','MAE','RMSE']].to_string(index=False))
print("\n✅ LSTM COMPLETED")
