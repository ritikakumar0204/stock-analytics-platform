# !pip install --quiet xgboost tensorflow

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import google.auth
from google.colab import auth
auth.authenticate_user()
from google.cloud import bigquery

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from statsmodels.tsa.arima.model import ARIMA

PROJECT = "sustained-pod-369913"
DATASET = "stock_forecasting"
MODEL_INPUT_TABLE = "model_input"         # table with technical features + targets
PREDICTIONS_TABLE = "predictions"        # table to write final predictions
MAX_TICKERS = 100  # set to an integer to limit the number for testing (e.g., 20). None = all.
N_FOLDS = 9          # number of OOF folds (expanding-window)
VAL_SPACING_DAYS = 14  # spacing between validation dates (approx every ~2 weeks)
DO_LSTM = True
LOOKBACK = 60        # LSTM lookback length (days)
LSTM_EPOCHS = 20
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

FEATURE_COLS = [
 'log_return','return_1d',
 'ma_5','ma_10','ma_20','ma_50','ema_12','ema_26','ma_10_minus_ma_50','macd','macd_signal',
 'rsi_14','atr_14',
 'vol_10','vol_20','vol_60',
 'mom_1m','mom_3m','mom_6m','mom_12m',
 'avg_volume_20','log_market_cap','day_of_week','is_month_end',
 'mom_3m_z','mom_3m_rankpct','vol_20_z','vol_20_rankpct'
]
TARGET_1D = 'target_return_1d'
TARGET_5D = 'target_return_5d'

print("Loading model_input from BigQuery...")
query = f"SELECT * FROM `{PROJECT}.{DATASET}.{MODEL_INPUT_TABLE}`"
df = client.query(query).to_dataframe()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['ticker','date']).reset_index(drop=True)
print("Rows loaded:", len(df), "unique tickers:", df['ticker'].nunique())

tickers

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['ticker','date']).reset_index(drop=True)

# limit tickers for testing
if MAX_TICKERS is not None:
    # tickers = sorted(df['ticker'].unique())[:MAX_TICKERS]
    df = df[df['ticker'].isin(tickers)].reset_index(drop=True)
else:
    tickers = sorted(df['ticker'].unique())

# Ensure targets exist
if TARGET_1D not in df.columns or TARGET_5D not in df.columns:
    raise ValueError("target columns missing. Ensure model_input contains target_return_1d and target_return_5d")

df.tail()

df['ticker'].unique()

df = df.dropna(subset=[TARGET_5D]).reset_index(drop=True)

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def safe_arima_forecast(series_prices, horizon=1, min_obs=80, clip_pct=0.5):

    if not isinstance(series_prices, pd.Series):
        try:
            series_prices = pd.Series(series_prices)
        except Exception:
            return np.nan

    # Drop NaNs at tail/head
    series = series_prices.dropna().copy()
    if series.empty or len(series) < min_obs:
        return np.nan
    try:
        if not isinstance(series.index, pd.DatetimeIndex):
            try:
                series.index = pd.to_datetime(series.index)
            except Exception:
                series = series.reset_index(drop=True)
                return _arima_forecast_numpy(series, horizon, clip_pct)
    except Exception:
        return np.nan

    if series.index.freq is None:
        inferred = pd.infer_freq(series.index)
        if inferred is None:
            # imposes business-day frequency (trading days) by reindexing and forward-filling
            # create business day index from min to max
            bidx = pd.date_range(start=series.index.min(), end=series.index.max(), freq='B')
            series = series.reindex(bidx)
            # forward fill any missing trading days - assuming point-in-time alignment
            series = series.fillna(method='ffill').fillna(method='bfill')
            # if still NaNs (all NaN), fallback to numpy method
            if series.isna().all():
                return _arima_forecast_numpy(series_prices.dropna(), horizon, clip_pct)
        else:
            # set inferred freq
            series = series.asfreq(inferred).fillna(method='ffill').fillna(method='bfill')

    # try to fit ARIMA on the date-indexed series
    try:
        model = ARIMA(series, order=(1,1,1))
        res = model.fit(method_kwargs={"warn_convergence": False})
        fc = res.forecast(steps=horizon)
        # fc could be a pd.Series or np.ndarray: take last horizon value
        if isinstance(fc, (pd.Series, pd.DataFrame)):
            pred_close = float(fc.iloc[-1])
        else:
            pred_close = float(fc[-1])
        last_close = float(series.iloc[-1])
        pred_ret = pred_close / last_close - 1.0
        # clip insane predictions
        pred_ret = float(np.clip(pred_ret, -clip_pct, clip_pct))
        return pred_ret
    except Exception as e:
        return _arima_forecast_numpy(series.values, horizon, clip_pct)


def _arima_forecast_numpy(arr, horizon, clip_pct):
    """Fit ARIMA on numpy ndarray values; returns predicted return or nan."""
    try:
        s = pd.Series(np.asarray(arr).astype(float))
        model = ARIMA(s, order=(1,1,1))
        res = model.fit(method_kwargs={"warn_convergence": False})
        fc = res.forecast(steps=horizon)
        pred_close = float(fc[-1]) if isinstance(fc, (list, np.ndarray)) else float(fc.iloc[-1])
        last_close = float(s.iloc[-1])
        pred_ret = pred_close / last_close - 1.0
        pred_ret = float(np.clip(pred_ret, -clip_pct, clip_pct))
        return pred_ret
    except Exception:
        return np.nan


def train_xgb_reg(train_X, train_y):
    dtrain = xgb.DMatrix(train_X, label=train_y)
    params = {"objective":"reg:squarederror","eta":0.05,"max_depth":5,"subsample":0.8,"colsample_bytree":0.6,"seed":SEED,"verbosity":0}
    model = xgb.train(params, dtrain, num_boost_round=800, verbose_eval=False)
    return model

def build_lstm(n_features, units=64):
    model = keras.models.Sequential([
        keras.layers.Input(shape=(LOOKBACK, n_features)),
        keras.layers.LSTM(units, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

oof_records = []
fold_summaries = []
print("Starting OOF generation across cutoffs. Note: this may take time for many tickers.")

from dateutil.relativedelta import relativedelta

def generate_expanding_folds(df,
                             train_start=None,
                             min_train_days=365*2,
                             val_months=6,
                             step_months=6,
                             final_dev_end=None,
                             max_horizon_days=5):

    df = df.sort_values("date").copy()
    df['date'] = pd.to_datetime(df['date'])
    data_min = df['date'].min()
    data_max = df['date'].max()

    if train_start is None:
        train_start = data_min
    train_start = pd.to_datetime(train_start)

    if final_dev_end is None:
        final_dev_end = data_max - pd.Timedelta(days=max_horizon_days)
    else:
        final_dev_end = pd.to_datetime(final_dev_end)

    # initial train_end: must be at least min_train_days after train_start
    train_end = train_start + pd.Timedelta(days=min_train_days)

    folds = []
    val_window = relativedelta(months=val_months)
    step = relativedelta(months=step_months)

    # Expand until train_end + val_window <= final_dev_end
    while train_end + val_window <= final_dev_end:
        val_start = train_end + pd.Timedelta(days=1)
        val_end = val_start + val_window - pd.Timedelta(days=1)   # inclusive

        train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)].copy()
        val_df   = df[(df['date'] >= val_start) & (df['date'] <= val_end)].copy()

        if not train_df.empty and not val_df.empty:
            meta = {"train_start": train_start, "train_end": train_end, "val_start": val_start, "val_end": val_end}
            folds.append((train_df, val_df, meta))

        # advance the train_end
        train_end = train_end + step

    return folds

def safe_median_fill(X_train_df, X_val_df):
    """
    Produce a value for filling NaNs in val using medians computed from X_train_df columns.
    Returns filled val DataFrame.
    """
    med = X_train_df.median()
    # if any medians NaN, fill with zero (last resort)
    med = med.fillna(0)
    return X_val_df.fillna(med)

# --- parameters you can tune ---
MIN_TRAIN_DAYS = 365*2     # start first fold after 2 years of train data
VAL_MONTHS = 6             # validation window size
STEP_MONTHS = 6            # how far to expand train_end for next fold
FINAL_DEV_END = None       # None -> will use df.max() - max_horizon_days
MAX_HORIZON_DAYS = 5       # for 5d target, ensure we don't pick val windows too near the end

df['date'] = pd.to_datetime(df['date'])

required_vars = ['FEATURE_COLS','TARGET_1D','TARGET_5D']
for v in required_vars:
    if v not in globals():
        raise RuntimeError(f"Required variable {v} not found in scope. Please define {v} before running folds.")

# prepare folds (this covers both 20 and 200 tickers; selection of tickers is driven by df contents)
folds = generate_expanding_folds(df,
                                train_start=None,
                                min_train_days=MIN_TRAIN_DAYS,
                                val_months=VAL_MONTHS,
                                step_months=STEP_MONTHS,
                                final_dev_end=FINAL_DEV_END,
                                max_horizon_days=MAX_HORIZON_DAYS)

oof_records = []   # reset / initialize
print(f"Generated {len(folds)} folds for walk-forward CV")

# Loop folds (expanding window)
for fold_idx, (train_df, val_df, meta) in enumerate(folds):
    print(f"\nFOLD {fold_idx}  TRAIN {meta['train_start'].date()} -> {meta['train_end'].date()}  VAL {meta['val_start'].date()} -> {meta['val_end'].date()}")

    # fold metadata for later saving
    fold_meta = {
        'fold_idx': int(fold_idx),
        'train_start': meta['train_start'],
        'train_end': meta['train_end'],
        'val_start': meta['val_start'],
        'val_end': meta['val_end'],
        'n_train_rows': len(train_df),
        'n_val_rows': len(val_df)
    }

    # drop rows in train/val that don't have target labels
    # important: for 5d target, ensure val rows do not include those near dataset end (generator already guarded that)
    train_df = train_df.dropna(subset=[TARGET_1D, TARGET_5D], how='all')  # keep rows that have at least one target (both preferred)
    val_df   = val_df.dropna(subset=[TARGET_1D, TARGET_5D], how='all')    # val rows with at least one target

    if train_df.empty or val_df.empty:
        print("  skipping fold: empty train or val after dropping missing targets")
        continue

    # ---- per-ticker ARIMA (1d & 5d) ----
    arima1 = {}; arima5 = {};
    for t in val_df['ticker'].unique():
        tr = train_df[train_df['ticker']==t].sort_values('date')
        if len(tr) < 10:   # minimal train length safeguard
            arima1[t] = np.nan; arima5[t] = np.nan;
            continue
        prices = tr['adj_close']
        # your arima_forecast/prophet_forecast should accept series and dates; keep as-is
        try:
            arima1[t] = safe_arima_forecast(prices, horizon=1)
        except Exception as e:
            print(f"  arima1 failed for {t}: {e}"); arima1[t] = np.nan
        try:
            arima5[t] = safe_arima_forecast(prices, horizon=5)
        except Exception as e:
            print(f"  arima5 failed for {t}: {e}"); arima5[t] = np.nan

    # ---- XGBoost pooled: train two separate models for 1d and 5d ----
    X_train = train_df[FEATURE_COLS].copy()
    y1_train = train_df[TARGET_1D].copy()
    y5_train = train_df[TARGET_5D].copy()

    mask1 = y1_train.notna() & X_train.notna().all(axis=1)
    mask5 = y5_train.notna() & X_train.notna().all(axis=1)

    xgb1_model = None; xgb5_model = None
    if mask1.sum() > 100:
        xgb1_model = train_xgb_reg(X_train[mask1].values, y1_train[mask1].values)
    if mask5.sum() > 100:
        xgb5_model = train_xgb_reg(X_train[mask5].values, y5_train[mask5].values)

    # XGB predictions for val (fill NaNs in val using train median)
    xgb1_preds = {}
    xgb5_preds = {}
    if xgb1_model is not None:
        valX = safe_median_fill(X_train[mask1], val_df[FEATURE_COLS].copy())
        xgb1_preds_values = xgb1_model.predict(xgb.DMatrix(valX.values))
        for t,p in zip(val_df['ticker'].values, xgb1_preds_values): xgb1_preds[t]=float(p)
    else:
        for t in val_df['ticker'].unique(): xgb1_preds[t]=np.nan

    if xgb5_model is not None:
        valX5 = safe_median_fill(X_train[mask5], val_df[FEATURE_COLS].copy())
        xgb5_preds_values = xgb5_model.predict(xgb.DMatrix(valX5.values))
        for t,p in zip(val_df['ticker'].values, xgb5_preds_values): xgb5_preds[t]=float(p)
    else:
        for t in val_df['ticker'].unique(): xgb5_preds[t]=np.nan

    # ---- LSTM pooled (optional) ----
    lstm1_preds = {}; lstm5_preds = {}
    if DO_LSTM:
        seqs = []
        # build sequences using only train_df (pooled across tickers)
        for t in train_df['ticker'].unique():
            df_t = train_df[train_df['ticker']==t].sort_values('date')
            if len(df_t) < LOOKBACK + 1:
                continue
            feat_arr = df_t[FEATURE_COLS].fillna(method='ffill').fillna(method='bfill').fillna(0).values
            y1 = df_t[TARGET_1D].values
            y5 = df_t[TARGET_5D].values
            for i in range(LOOKBACK, len(df_t)):
                if np.isnan(y1[i]) or np.isnan(y5[i]):
                    continue
                Xseq = feat_arr[i-LOOKBACK:i]
                seqs.append((Xseq, y1[i], y5[i]))

        if len(seqs) >= 200:  # minimal size
            X = np.stack([s[0] for s in seqs])
            y1_arr = np.array([s[1] for s in seqs])
            y5_arr = np.array([s[2] for s in seqs])
            # split 80/20
            idx_split = int(len(X)*0.8)
            Xtr, Xval_internal = X[:idx_split], X[idx_split:]
            y1tr, y1val_internal = y1_arr[:idx_split], y1_arr[idx_split:]
            y5tr, y5val_internal = y5_arr[:idx_split], y5_arr[idx_split:]
            n_features = X.shape[2]
            scaler = StandardScaler()
            Xtr_flat = Xtr.reshape(-1, n_features)
            Xval_flat = Xval_internal.reshape(-1, n_features)
            scaler.fit(Xtr_flat)
            Xtr_s = scaler.transform(Xtr_flat).reshape(Xtr.shape)
            Xval_s = scaler.transform(Xval_flat).reshape(Xval_internal.shape)
            # model for 1d
            model1 = build_lstm(n_features, units=32)
            es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
            model1.fit(Xtr_s, y1tr, validation_data=(Xval_s,y1val_internal), epochs=LSTM_EPOCHS, batch_size=64, callbacks=[es], verbose=0)
            # model for 5d
            model5 = build_lstm(n_features, units=32)
            model5.fit(Xtr_s, y5tr, validation_data=(Xval_s,y5val_internal), epochs=LSTM_EPOCHS, batch_size=64, callbacks=[es], verbose=0)

            # predict for each val ticker/date using sequence up to last train date per ticker
            for t in val_df['ticker'].unique():
                df_t = train_df[train_df['ticker']==t].sort_values('date')
                if len(df_t) < LOOKBACK:
                    lstm1_preds[t] = np.nan; lstm5_preds[t] = np.nan; continue
                seq = df_t[FEATURE_COLS].fillna(method='ffill').fillna(method='bfill').fillna(0).values[-LOOKBACK:]
                seq_s = scaler.transform(seq).reshape(1,LOOKBACK,-1)
                lstm1_preds[t] = float(model1.predict(seq_s, verbose=0)[0,0])
                lstm5_preds[t] = float(model5.predict(seq_s, verbose=0)[0,0])
        else:
            for t in val_df['ticker'].unique(): lstm1_preds[t]=np.nan; lstm5_preds[t]=np.nan
    else:
        for t in val_df['ticker'].unique(): lstm1_preds[t]=np.nan; lstm5_preds[t]=np.nan

    # ---- collect OOF per val row ----
    for _, vr in val_df.iterrows():
        t = vr['ticker']
        rec = {
              'fold_idx': fold_idx,
              'train_start': fold_meta['train_start'],
              'train_end': fold_meta['train_end'],
              'val_start': fold_meta['val_start'],
              'val_end': fold_meta['val_end'],
              'ticker': t,
              'date': vr['date'],
              'true1': float(vr[TARGET_1D]) if pd.notna(vr[TARGET_1D]) else np.nan,
              'true5': float(vr[TARGET_5D]) if pd.notna(vr[TARGET_5D]) else np.nan,
              'arima_1d': float(arima1.get(t, np.nan)),
              'xgb_1d': float(xgb1_preds.get(t, np.nan)),
              'lstm_1d': float(lstm1_preds.get(t, np.nan)) if DO_LSTM else np.nan,
              'arima_5d': float(arima5.get(t, np.nan)),
              'xgb_5d': float(xgb5_preds.get(t, np.nan)),
              'lstm_5d': float(lstm5_preds.get(t, np.nan)) if DO_LSTM else np.nan
              }
        oof_records.append(rec)

print("OOF generation finished. OOF rows:", len(oof_records))

oof = pd.DataFrame(oof_records)
# drop rows with all base preds missing
base_cols = ['arima_1d','xgb_1d','lstm_1d','arima_5d','xgb_5d','lstm_5d']
oof = oof.dropna(subset=base_cols, how='all').reset_index(drop=True)
# fill NaNs in base preds with 0 (or could use median)
oof_f = oof.copy()
oof_f[base_cols] = oof_f[base_cols].fillna(0)

oof_f

import sklearn.metrics as skm
import scipy.stats as ss
import numpy as np
import pandas as pd

fold_oof_df = pd.DataFrame([r for r in oof_records if r['fold_idx'] == fold_idx])

# save per-fold raw OOF rows for debugging
fold_oof_df.to_csv(f"fold_oof_fold{fold_idx}.csv", index=False)

# drop rows with no true values
fold_oof_valid_1d = fold_oof_df.dropna(subset=['true1']).copy()
fold_oof_valid_5d = fold_oof_df.dropna(subset=['true5']).copy()

metrics = {}

# helper to compute metrics robustly
def compute_metrics(y_true, y_pred):
    """Return dict: mae, rmse, dir_acc, spearman_ic"""
    if len(y_true) == 0:
        return {'mae':np.nan,'rmse':np.nan,'dir':np.nan,'ic':np.nan}
    # mask to remove NaN preds
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {'mae':np.nan,'rmse':np.nan,'dir':np.nan,'ic':np.nan}
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = skm.mean_absolute_error(yt, yp)
    rmse = np.sqrt(skm.mean_squared_error(yt, yp))
    dir_acc = np.mean((np.sign(yp) == np.sign(yt)).astype(float))
    try:
        ic = ss.spearmanr(yp, yt).correlation
    except Exception:
        ic = np.nan
    return {'mae':mae,'rmse':rmse,'dir':dir_acc,'ic':ic}

# list of base models per horizon
base_models_1d = ['arima_1d','xgb_1d','lstm_1d']
base_models_5d = ['arima_5d','xgb_5d','lstm_5d']

# compute metrics for 1d
if len(fold_oof_valid_1d):
    y1 = fold_oof_valid_1d['true1'].values
    # per-model metrics
    for m in base_models_1d:
        preds = fold_oof_valid_1d.get(m, pd.Series(np.nan, index=fold_oof_valid_1d.index)).values
        res = compute_metrics(y1, preds)
        metrics[f'{m}_mae'] = res['mae']
        metrics[f'{m}_rmse'] = res['rmse']
        metrics[f'{m}_dir'] = res['dir']
        metrics[f'{m}_ic'] = res['ic']

    # ensemble (simple mean of available models) for 1d
    ensemble_preds_1d = fold_oof_valid_1d[base_models_1d].fillna(0).mean(axis=1).values
    ens_res = compute_metrics(y1, ensemble_preds_1d)
    metrics['ensemble_1d_mae'] = ens_res['mae']
    metrics['ensemble_1d_rmse'] = ens_res['rmse']
    metrics['ensemble_1d_dir'] = ens_res['dir']
    metrics['ensemble_1d_ic'] = ens_res['ic']
else:
    # fill NaNs if no valid rows
    for m in base_models_1d:
        metrics[f'{m}_mae'] = metrics[f'{m}_rmse'] = metrics[f'{m}_dir'] = metrics[f'{m}_ic'] = np.nan
    metrics['ensemble_1d_mae'] = metrics['ensemble_1d_rmse'] = metrics['ensemble_1d_dir'] = metrics['ensemble_1d_ic'] = np.nan

# compute metrics for 5d
if len(fold_oof_valid_5d):
    y5 = fold_oof_valid_5d['true5'].values
    for m in base_models_5d:
        preds = fold_oof_valid_5d.get(m, pd.Series(np.nan, index=fold_oof_valid_5d.index)).values
        res = compute_metrics(y5, preds)
        metrics[f'{m}_mae'] = res['mae']
        metrics[f'{m}_rmse'] = res['rmse']
        metrics[f'{m}_dir'] = res['dir']
        metrics[f'{m}_ic'] = res['ic']

    # ensemble for 5d
    ensemble_preds_5d = fold_oof_valid_5d[base_models_5d].fillna(0).mean(axis=1).values
    ens_res5 = compute_metrics(y5, ensemble_preds_5d)
    metrics['ensemble_5d_mae'] = ens_res5['mae']
    metrics['ensemble_5d_rmse'] = ens_res5['rmse']
    metrics['ensemble_5d_dir'] = ens_res5['dir']
    metrics['ensemble_5d_ic'] = ens_res5['ic']
else:
    for m in base_models_5d:
        metrics[f'{m}_mae'] = metrics[f'{m}_rmse'] = metrics[f'{m}_dir'] = metrics[f'{m}_ic'] = np.nan
    metrics['ensemble_5d_mae'] = metrics['ensemble_5d_rmse'] = metrics['ensemble_5d_dir'] = metrics['ensemble_5d_ic'] = np.nan

# combine metrics + fold meta and append
fold_summary = {**fold_meta, **metrics}
fold_summaries.append(fold_summary)

# persist per-fold summary and fold OOF for auditing
pd.DataFrame([fold_summary]).to_csv(f"fold_summary_{fold_idx}.csv", index=False)
fold_oof_df.to_csv(f"fold_oof_fold{fold_idx}.csv", index=False)

# quick print summary
print(f"Saved fold {fold_idx} summary — {fold_meta['n_train_rows']} train rows, {fold_meta['n_val_rows']} val rows.")
print("1d ensemble MAE:", metrics.get('ensemble_1d_mae'), "5d ensemble MAE:", metrics.get('ensemble_5d_mae'))

# === Fit meta models for 5d and 1d, compute residual stds and add oof meta preds ===
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error

base_cols = ['arima_1d','xgb_1d','lstm_1d','arima_5d','xgb_5d','lstm_5d']

# --- meta for 5d
X_meta5 = oof_f[base_cols].fillna(0).values
y_meta5 = oof_f['true5'].values

meta5 = RidgeCV(alphas=[0.01,0.1,1.0,10.0], cv=None)
meta5.fit(X_meta5, y_meta5)
oof_f['meta_pred_5d'] = meta5.predict(X_meta5)
mae_meta5 = mean_absolute_error(y_meta5, oof_f['meta_pred_5d'])
ic_meta5 = np.corrcoef(y_meta5, oof_f['meta_pred_5d'])[0,1]
print("Meta(5d) MAE:", mae_meta5, "IC:", ic_meta5)
print("Meta(5d) weights:", dict(zip(base_cols, meta5.coef_.round(6))))

meta5_resid_std = np.nanstd(y_meta5 - oof_f['meta_pred_5d'])

# --- meta for 1d
X_meta1 = oof_f[base_cols].fillna(0).values
y_meta1 = oof_f['true1'].values

# For rows where true1 is NaN, RidgeCV still fits but we'd prefer fitting on non-NaN rows
mask_y1 = ~np.isnan(y_meta1)
if mask_y1.sum() > 20:
    meta1 = RidgeCV(alphas=[0.01,0.1,1.0,10.0], cv=None)
    meta1.fit(X_meta1[mask_y1], y_meta1[mask_y1])
    oof_f['meta_pred_1d'] = meta1.predict(X_meta1)
    mae_meta1 = mean_absolute_error(y_meta1[mask_y1], oof_f.loc[mask_y1, 'meta_pred_1d'])
    ic_meta1 = np.corrcoef(y_meta1[mask_y1], oof_f.loc[mask_y1, 'meta_pred_1d'])[0,1]
    print("Meta(1d) MAE:", mae_meta1, "IC:", ic_meta1)
    print("Meta(1d) weights:", dict(zip(base_cols, meta1.coef_.round(6))))
    meta1_resid_std = np.nanstd(y_meta1[mask_y1] - oof_f.loc[mask_y1, 'meta_pred_1d'])
else:
    # not enough data to train meta1 robustly
    meta1 = None
    oof_f['meta_pred_1d'] = np.nan
    meta1_resid_std = np.nan
    print("Not enough non-NaN y_meta1 to fit meta1 (need >20). meta_pred_1d filled with NaN.")

# === Retrain base pooled models on full history (unchanged) ===
print("Retraining base models on full history and producing final predictions for next day...")
next_day = datetime.today() + pd.Timedelta(days=1)
full = df.copy()

full_X = full[FEATURE_COLS].copy()
full_y1 = full[TARGET_1D].copy()
full_y5 = full[TARGET_5D].copy()
mask1 = full_y1.notna() & full_X.notna().all(axis=1)
mask5 = full_y5.notna() & full_X.notna().all(axis=1)
xgb1_full = train_xgb_reg(full_X[mask1].values, full_y1[mask1].values) if mask1.sum() > 100 else None
xgb5_full = train_xgb_reg(full_X[mask5].values, full_y5[mask5].values) if mask5.sum() > 100 else None

# === Retrain pooled LSTM on full history ===
lstm1_full = None; lstm5_full = None; lstm_scaler = None
if DO_LSTM:
    seqs_full = []
    for t in full['ticker'].unique():
        df_t = full[full['ticker']==t].sort_values('date')
        if len(df_t) < LOOKBACK+1: continue
        feats = df_t[FEATURE_COLS].fillna(method='ffill').fillna(method='bfill').fillna(0).values
        y1arr = df_t[TARGET_1D].values; y5arr = df_t[TARGET_5D].values
        for i in range(LOOKBACK, len(df_t)):
            if np.isnan(y1arr[i]) or np.isnan(y5arr[i]): continue
            seqs_full.append((feats[i-LOOKBACK:i], y1arr[i], y5arr[i]))
    if len(seqs_full) >= 300:
        Xf = np.stack([s[0] for s in seqs_full])
        y1f = np.array([s[1] for s in seqs_full]); y5f = np.array([s[2] for s in seqs_full])
        nfeat = Xf.shape[2]
        lstm_scaler = StandardScaler()
        Xf_flat = Xf.reshape(-1, nfeat)
        lstm_scaler.fit(Xf_flat)
        Xf_s = lstm_scaler.transform(Xf_flat).reshape(Xf.shape)
        lstm1_full = build_lstm(nfeat, units=32)
        lstm1_full.fit(Xf_s, y1f, epochs=8, batch_size=64, verbose=0)
        lstm5_full = build_lstm(nfeat, units=32)
        lstm5_full.fit(Xf_s, y5f, epochs=8, batch_size=64, verbose=0)

# === Final prediction loop: compute full set of base preds produce meta preds for 1d and 5d, and compute confidences ===
final_rows = []
for t in sorted(full['ticker'].unique()):
    last_row = full[full['ticker']==t].sort_values('date').iloc[-1]
    # Price series for per-ticker forecasts
    tr = full[full['ticker']==t].sort_values('date')
    prices = tr['adj_close']

    # ARIMA forecasts
    ar1 = safe_arima_forecast(prices, horizon=1)
    ar5 = safe_arima_forecast(prices, horizon=5)

    # XGB predictions
    x1 = np.nan; x5 = np.nan
    feat_row = last_row[FEATURE_COLS].copy().to_frame().T
    if xgb1_full is not None:
        feat_row1 = feat_row.fillna(full_X[mask1].median())
        x1 = float(xgb1_full.predict(xgb.DMatrix(feat_row1.values)))
    if xgb5_full is not None:
        feat_row5 = feat_row.fillna(full_X[mask5].median())
        x5 = float(xgb5_full.predict(xgb.DMatrix(feat_row5.values)))

    # LSTM predictions on last LOOKBACK rows
    l1 = np.nan; l5 = np.nan
    df_t = full[full['ticker']==t].sort_values('date')
    if DO_LSTM and lstm1_full is not None and len(df_t) >= LOOKBACK:
        seq = df_t[FEATURE_COLS].fillna(method='ffill').fillna(method='bfill').fillna(0).values[-LOOKBACK:]
        seq_s = lstm_scaler.transform(seq).reshape(1,LOOKBACK,-1)
        l1 = float(lstm1_full.predict(seq_s, verbose=0)[0,0]) if lstm1_full is not None else np.nan
        l5 = float(lstm5_full.predict(seq_s, verbose=0)[0,0]) if lstm5_full is not None else np.nan

    # Build base vector
    base_vec = np.array([[
        np.nan_to_num(ar1), np.nan_to_num(x1), np.nan_to_num(l1),
        np.nan_to_num(ar5), np.nan_to_num(x5), np.nan_to_num(l5)
    ]])

    # Meta predictions: 5d and 1d (use trained meta5 and meta1 if available)
    meta_pred_5d = float(meta5.predict(base_vec)[0]) if meta5 is not None else np.nan
    meta_pred_1d = float(meta1.predict(base_vec)[0]) if (meta1 is not None) else np.nan

    # --- Confidence calculations for 5d ---
    # use only 5d-capable base preds (ar5,x5,l5), fallback to corresponding 1d if 5d missing
    preds_5d = np.array([ar5, x5, l5], dtype=float)
    # fallback: if a specific 5d is nan, try corresponding 1d
    for i in range(3):
        if np.isnan(preds_5d[i]):
            preds_5d[i] = np.nan  # keeping NaN;
    # compute dispersion: ignore NaNs
    valid5 = preds_5d[~np.isnan(preds_5d)]
    base_std_5d = np.nanstd(valid5) if valid5.size>0 else np.nan
    base_std_conf_5d = 1.0 / (1.0 + (base_std_5d if not np.isnan(base_std_5d) else 1.0))

    # sign agreement for 5d: fraction of models that agree with meta_pred_5d sign
    sign_votes_5 = []
    for val in preds_5d:
        if np.isnan(val):
            continue
        sign_votes_5.append(np.sign(val) if np.sign(val)!=0 else 1.0)
    if len(sign_votes_5)==0:
        sign_agreement_5 = 0.0
    else:
        meta_sign5 = np.sign(meta_pred_5d) if meta_pred_5d != 0 else 1.0
        sign_agreement_5 = np.mean([1.0 if s == meta_sign5 else 0.0 for s in sign_votes_5])

    meta_consistency_5 = 1.0 / (1.0 + (meta5_resid_std if not np.isnan(meta5_resid_std) else 1.0))

    confidence_5d = 0.5 * base_std_conf_5d + 0.4 * sign_agreement_5 + 0.1 * meta_consistency_5
    confidence_5d = float(np.clip(confidence_5d, 0.0, 1.0))

    # --- Confidence calculations for 1d ---
    preds_1d = np.array([ar1, x1, l1], dtype=float)
    valid1 = preds_1d[~np.isnan(preds_1d)]
    base_std_1d = np.nanstd(valid1) if valid1.size>0 else np.nan
    base_std_conf_1d = 1.0 / (1.0 + (base_std_1d if not np.isnan(base_std_1d) else 1.0))

    sign_votes_1 = []
    for val in preds_1d:
        if np.isnan(val):
            continue
        sign_votes_1.append(np.sign(val) if np.sign(val)!=0 else 1.0)
    if len(sign_votes_1)==0:
        sign_agreement_1 = 0.0
    else:
        meta_sign1 = np.sign(meta_pred_1d) if (meta_pred_1d is not None and not np.isnan(meta_pred_1d) and meta_pred_1d != 0) else 1.0
        sign_agreement_1 = np.mean([1.0 if s == meta_sign1 else 0.0 for s in sign_votes_1])

    meta_consistency_1 = 1.0 / (1.0 + (meta1_resid_std if not np.isnan(meta1_resid_std) else 1.0))

    confidence_1d = 0.5 * base_std_conf_1d + 0.4 * sign_agreement_1 + 0.1 * meta_consistency_1
    confidence_1d = float(np.clip(confidence_1d, 0.0, 1.0))

    final_rows.append({
        'ticker': t,
        'date': next_day.date(),
        'arima_1d': ar1, 'xgb_1d': x1, 'lstm_1d': l1,
        'arima_5d': ar5, 'xgb_5d': x5, 'lstm_5d': l5,
        'meta_pred_5d': meta_pred_5d,
        'meta_pred_1d': meta_pred_1d,
        'confidence_5d': confidence_5d,
        'confidence_1d': confidence_1d,
        'created_at': datetime.utcnow()
    })

final_preds_df = pd.DataFrame(final_rows)
print("Final predictions prepared:", len(final_preds_df))

final_preds_df.head(2)

import pandas as pd
import matplotlib.pyplot as plt

fs = pd.read_csv("/content/fold_summary_6.csv", parse_dates=['train_start','train_end','val_start','val_end'])
# example: plot per-fold ensemble MAE vs fold_idx
plt.plot(fs['fold_idx'], fs['ensemble_mae'], marker='o', label='ensemble_mae')
plt.plot(fs['fold_idx'], fs['xgb_1d_mae'], marker='x', label='xgb_1d_mae')
plt.plot(fs['fold_idx'], fs['prophet_1d_mae'], marker='s', label='prophet_1d_mae')
plt.legend(); plt.xlabel('fold_idx'); plt.ylabel('MAE'); plt.title('Per-fold MAE')
plt.show()

# compute trend (is MAE decreasing across folds?)
from scipy.stats import linregress
slope, intercept, r, p, se = linregress(fs['fold_idx'], fs['ensemble_mae'])
print("Ensemble MAE slope per fold:", slope, "p-value:", p)

query = f"SELECT * FROM `{BQ_PROJECT}.{BQ_DATASET}.{PRICES_TABLE}` ORDER BY ticker, date"
prices = client.query(query).to_dataframe()
prices

prices2 = prices.copy()
oof2 = oof_f.copy()
final2 = final_preds_df.copy()
folds2 = pd.DataFrame(fold_summaries).copy() if not isinstance(fold_summaries, pd.DataFrame) else fold_summaries.copy()

final2['date'] = prices2['date'].max()
final2.head()

dashboard_df = prices2.merge(
    final2.assign(date=pd.to_datetime(final2['date'])),
    on=['ticker', 'date'],
    how='left'
)
dashboard_df.tail()

dashboard_df_final = dashboard_df[dashboard_df['ticker'].isin(tickers)]
dashboard_df_final.tail()

dashboard_df_final.to_csv("dashboard_df_final.csv", index=False)

prices2.to_csv("prices2.csv", index=False)
oof2.to_csv("oof2.csv", index=False)
final2.to_csv("final2.csv", index=False)
folds2.to_csv("folds2.csv", index=False)

# Combine prices, oof_f, final_preds_df, fold_summaries into one CSV
import pandas as pd
import numpy as np

# Expect these DataFrames in memory: prices, oof_f, final_preds_df, fold_summaries
prices2 = prices.copy()
oof2 = oof_f.copy()
final2 = final_preds_df.copy()
folds2 = pd.DataFrame(fold_summaries).copy() if not isinstance(fold_summaries, pd.DataFrame) else fold_summaries.copy()

# Normalize date columns
prices2['date'] = pd.to_datetime(prices2['date'])
oof2['date'] = pd.to_datetime(oof2['date'])
final2['date'] = pd.to_datetime(final2['date'])

# Ensure truth columns in OOF are named 'true1' and 'true5' if alternatives exist
if 'true1' not in oof2.columns:
    if 'target_return_1d' in oof2.columns:
        oof2 = oof2.rename(columns={'target_return_1d':'true1'})
if 'true5' not in oof2.columns:
    if 'target_return_5d' in oof2.columns:
        oof2 = oof2.rename(columns={'target_return_5d':'true5'})

# Normalize final preds: prefer 'pred_1d' if present or leave existing prediction columns
if 'pred_1d' not in final2.columns:
    for alt in ['pred','prediction','prediction_1d','meta_pred_5d','model_pred']:
        if alt in final2.columns:
            final2 = final2.rename(columns={alt:'pred_1d'})
            break

# Convert created_at in final predictions to datetime if present
if 'created_at' in final2.columns:
    final2['created_at'] = pd.to_datetime(final2['created_at'])

# Merge: prices LEFT JOIN oof (on ticker,date), then LEFT JOIN final preds, then LEFT JOIN fold metadata (if fold_idx present)
combined = prices2.merge(oof2, on=['ticker','date'], how='left', suffixes=('','_oof'))
combined = combined.merge(final2, on=['ticker','date'], how='left', suffixes=('','_final'))

# Attach fold metadata if available
if 'fold_idx' in combined.columns and 'fold_idx' in folds2.columns:
    # ensure fold date fields are datetime if present
    for c in ['train_start','train_end','val_start','val_end']:
        if c in folds2.columns:
            folds2[c] = pd.to_datetime(folds2[c])
    combined = combined.merge(folds2, on='fold_idx', how='left', suffixes=('','_fold'))

# Save single CSV file
combined.to_csv("dashboard_raw_combined.csv", index=False)

# Minimal confirmation print
print("Saved combined CSV: dashboard_raw_combined.csv")
print("Rows:", len(combined), "Columns:", len(combined.columns))
