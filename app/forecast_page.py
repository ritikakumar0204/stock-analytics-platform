import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import textwrap


@st.cache_data(show_spinner=False)
def load_historical_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns=str.title)
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close", "Date", "Ticker"])
    df = df.sort_values(["Ticker", "Date"])
    return df


@st.cache_data(show_spinner=False)
def load_returns_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

def run_prophet_forecast(hist_df: pd.DataFrame, ticker: str, forecast_horizon: int):
    prophet_impl = None
    impl_label = ""
    import_errs = []
    try:
        from prophet import Prophet  # type: ignore
        prophet_impl = Prophet
        impl_label = "prophet"
    except Exception as exc:
        import_errs.append(f"prophet import failed: {exc}")
        try:
            from fbprophet import Prophet  # type: ignore
            prophet_impl = Prophet
            impl_label = "fbprophet"
            st.info("Using fbprophet fallback.")
        except Exception as exc2:
            import_errs.append(f"fbprophet import failed: {exc2}")
            st.error("Failed to import Prophet/fbprophet. Install/repair the package and backend. "
                     + " | ".join(import_errs))
            return False

    data = hist_df[hist_df["Ticker"] == ticker][["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    data = data.sort_values("ds").dropna()

    if len(data) < 30:
        st.warning("Not enough data to run Prophet (need at least 30 rows).")
        return False

    train_size = int(len(data) * 0.8)
    val_size = int(len(data) * 0.1)
    if train_size == 0 or val_size == 0:
        st.warning("Not enough data to create train/validation/test splits.")
        return False

    train = data.iloc[:train_size]
    val = data.iloc[train_size : train_size + val_size]
    test = data.iloc[train_size + val_size :]

    try:
        m = prophet_impl(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(train)
    except Exception as exc:
        st.error(f"Failed to run {impl_label}: {exc}. Verify installation/backend (cmdstanpy or pystan).")
        return

    # Forecast for validation+test horizon
    horizon_hist = pd.concat([val, test])[["ds"]].reset_index(drop=True)

    last_date = data["ds"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon)
    future_df = pd.DataFrame({"ds": future_dates})

    future_all = pd.concat([horizon_hist, future_df], ignore_index=True)
    forecast = m.predict(future_all)
    preds = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    actuals = pd.concat([val, test]).reset_index(drop=True)
    results_hist = actuals.merge(preds, on="ds", how="left")
    results_future = preds[~preds["ds"].isin(results_hist["ds"])].copy()
    results_future["y"] = np.nan

    full_results = pd.concat([results_hist, results_future], ignore_index=True)
    full_results = full_results.sort_values("ds").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train["ds"], train["y"], label="Train", color="gray")
    ax.plot(val["ds"], val["y"], label="Validation", color="orange")
    ax.plot(test["ds"], test["y"], label="Test", color="green")
    ax.plot(full_results["ds"], full_results["yhat"], label="Forecast", color="blue", linewidth=1.5)
    ax.fill_between(
        full_results["ds"],
        full_results["yhat_lower"],
        full_results["yhat_upper"],
        color="blue",
        alpha=0.15,
        label="Forecast interval",
    )
    ax.set_title(f"{ticker} Prophet Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig)
    st.dataframe(full_results, use_container_width=True)
    return True


def run_xgb_forecast(hist_df: pd.DataFrame, ticker: str, forecast_horizon: int):
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        st.error(f"Failed to import xgboost: {exc}")
        return False

    data = hist_df[hist_df["Ticker"] == ticker][["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    data = data.sort_values("ds").dropna()

    # simple lag + calendar features
    data["y_lag1"] = data["y"].shift(1)
    data["dow"] = data["ds"].dt.dayofweek
    data["dom"] = data["ds"].dt.day
    data = data.dropna(subset=["y_lag1"])

    if len(data) < 30:
        st.warning("Not enough data to run XGBoost (need at least 30 rows).")
        return False

    train_size = int(len(data) * 0.8)
    val_size = int(len(data) * 0.1)
    if train_size == 0 or val_size == 0:
        st.warning("Not enough data to create train/validation/test splits.")
        return False

    train = data.iloc[:train_size]
    val = data.iloc[train_size : train_size + val_size]
    test = data.iloc[train_size + val_size :]

    feature_cols = ["y_lag1", "dow", "dom"]

    # Expanded grid search evaluated on the validation split
    param_grid = [
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8},
        {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.9},
        {"n_estimators": 600, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9},
        {"n_estimators": 800, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.85},
        {"n_estimators": 1000, "max_depth": 6, "learning_rate": 0.02, "subsample": 0.9},
    ]

    best_rmse = float("inf")
    best_params = None
    best_model = None

    try:
        for params in param_grid:
            model = XGBRegressor(
                **params,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
            )
            model.fit(train[feature_cols], train["y"])
            val_preds = model.predict(val[feature_cols])
            rmse = float(np.sqrt(np.mean((val_preds - val["y"]) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                best_model = model
    except Exception as exc:
        st.error(f"Failed during XGBoost grid search: {exc}")
        return False

    if best_model is None:
        st.error("No XGBoost model was trained successfully.")
        return False

    st.caption(f"Best XGBoost params: {best_params} (val RMSE: {best_rmse:.4f})")
    st.write(best_params)

    # Fit best model on train+val, then forecast val+test horizon
    train_val = pd.concat([train, val])
    best_model.fit(train_val[feature_cols], train_val["y"])

    pred_df = pd.concat([val, test])
    preds = best_model.predict(pred_df[feature_cols])
    results = pred_df[["ds", "y"]].copy()
    results["yhat"] = preds

    # Future recursive forecast
    last_date = data["ds"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon)
    future_df = pd.DataFrame({"ds": future_dates})
    future_df["y"] = np.nan
    future_df["dow"] = future_df["ds"].dt.dayofweek
    future_df["dom"] = future_df["ds"].dt.day

    prev_y = data.iloc[-1]["y"]
    yhat_values = []
    for i in range(forecast_horizon):
        row = future_df.iloc[i]
        X = pd.DataFrame(
            {
                "y_lag1": [prev_y],
                "dow": [row["dow"]],
                "dom": [row["dom"]],
            }
        )
        yhat = best_model.predict(X)[0]
        yhat_values.append(yhat)
        prev_y = yhat

    future_df["yhat"] = yhat_values

    # Uncertainty cone
    val_preds = best_model.predict(val[feature_cols])
    base_std = float(np.std(val_preds - val["y"])) if len(val) > 1 else 0.0
    z_outer = 2.58
    horizon_idx = np.arange(len(results) + len(future_df))
    widen = 1 + (horizon_idx / max(len(horizon_idx) - 1, 1)) * 1.5

    full_results = pd.concat([results, future_df], ignore_index=True)
    full_results = full_results.sort_values("ds").reset_index(drop=True)
    full_results["lower"] = full_results["yhat"] - z_outer * base_std * widen
    full_results["upper"] = full_results["yhat"] + z_outer * base_std * widen

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(train["ds"], train["y"], label="Train", color="gray")
    ax.plot(val["ds"], val["y"], label="Validation", color="orange")
    ax.plot(test["ds"], test["y"], label="Test", color="green")

    ax.plot(full_results["ds"], full_results["yhat"], label="Forecast (XGBoost)", color="blue", linewidth=2)

    if base_std > 0:
        ax.fill_between(
            full_results["ds"],
            full_results["lower"],
            full_results["upper"],
            color="#92B6FF",
            alpha=0.25,
        )

    ax.set_title(f"{ticker} XGBoost Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig)
    st.dataframe(full_results, use_container_width=True)
    return True


def run_arima_forecast(hist_df: pd.DataFrame, ticker: str, forecast_horizon: int):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as exc:
        st.error(f"Failed to import statsmodels ARIMA: {exc}")
        return False

    data_raw = (
        hist_df[hist_df["Ticker"] == ticker][["Date", "Close"]]
        .rename(columns={"Date": "ds", "Close": "y"})
        .sort_values("ds")
        .dropna()
        .reset_index(drop=True)
    )

    if len(data_raw) < 30:
        st.warning("Not enough data to run ARIMA (need at least 30 rows).")
        return False

    data = data_raw.copy()
    data["y_log"] = np.log(data["y"])
    data["y_diff"] = data["y_log"].diff()
    data_diff = data.dropna(subset=["y_diff"]).reset_index(drop=True)

    if len(data_diff) < 30:
        st.warning("Not enough data after differencing to run ARIMA (need at least 30 rows).")
        return False

    train_size = int(len(data_diff) * 0.8)
    val_size = int(len(data_diff) * 0.1)
    if train_size == 0 or val_size == 0:
        st.warning("Not enough data to create train/validation/test splits.")
        return False

    train = data_diff.iloc[:train_size]
    val = data_diff.iloc[train_size : train_size + val_size]
    test = data_diff.iloc[train_size + val_size :]

    candidate_orders = [(p, 1, q) for p in range(0, 4) for q in range(0, 4)]
    best_rmse = float("inf")
    best_order = None

    for order in candidate_orders:
        try:
            model = ARIMA(train["y_diff"], order=order)
            res = model.fit()
            val_pred = res.forecast(steps=len(val))
            rmse = float(np.sqrt(np.mean((val_pred - val["y_diff"].values) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_order = order
        except Exception:
            continue

    if best_order is None:
        st.error("No ARIMA configuration succeeded.")
        return False

    st.caption(f"Best ARIMA order: {best_order} (val RMSE: {best_rmse:.4f})")

    try:
        train_val_series = pd.concat([train["y_diff"], val["y_diff"]])
        res_hist = ARIMA(train_val_series, order=best_order).fit()
        steps_hist = len(val) + len(test)
        hist_fc = res_hist.get_forecast(steps=steps_hist)
        hist_mean = hist_fc.predicted_mean
        hist_ci = hist_fc.conf_int(alpha=0.05)
    except Exception as exc:
        st.error(f"Failed to train/forecast ARIMA on history: {exc}")
        return False

    base_log_hist = data_diff["y_log"].iloc[train_size - 1]
    hist_log = base_log_hist + hist_mean.cumsum()
    hist_pred = np.exp(hist_log)
    hist_lower = np.exp(base_log_hist + hist_ci.iloc[:, 0].cumsum())
    hist_upper = np.exp(base_log_hist + hist_ci.iloc[:, 1].cumsum())

    hist_ds = pd.concat([val["ds"], test["ds"]]).reset_index(drop=True)
    hist_actuals = pd.concat([val["y"], test["y"]]).reset_index(drop=True)
    results_hist = pd.DataFrame(
        {
            "ds": hist_ds,
            "y": hist_actuals,
            "yhat": hist_pred.values,
            "lower": hist_lower.values,
            "upper": hist_upper.values,
        }
    )

    try:
        res_full = ARIMA(data_diff["y_diff"], order=best_order).fit()
        future_fc = res_full.get_forecast(steps=forecast_horizon)
        future_mean = future_fc.predicted_mean
        future_ci = future_fc.conf_int(alpha=0.05)
    except Exception as exc:
        st.error(f"Failed to forecast future with ARIMA: {exc}")
        return False

    base_log_future = data_diff["y_log"].iloc[-1]
    future_log = base_log_future + future_mean.cumsum()
    future_pred = np.exp(future_log)
    future_lower = np.exp(base_log_future + future_ci.iloc[:, 0].cumsum())
    future_upper = np.exp(base_log_future + future_ci.iloc[:, 1].cumsum())

    last_date = data_raw["ds"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon)

    results_future = pd.DataFrame(
        {
            "ds": future_dates,
            "y": np.nan,
            "yhat": future_pred.values,
            "lower": future_lower.values,
            "upper": future_upper.values,
        }
    )

    full_results = pd.concat([results_hist, results_future], ignore_index=True).sort_values("ds").reset_index(drop=True)

    # Guard against inf/NaN outputs from unstable ARIMA fits
    full_results = full_results.replace([np.inf, -np.inf], np.nan)
    full_results = full_results.dropna(subset=["yhat", "lower", "upper", "ds"])
    if full_results.empty:
        st.error("ARIMA produced invalid forecasts (nan/inf). Try another model or different parameters.")
        return False

    # Ensure finite y-range for plotting
    y_candidates = []
    for series in [train["y"], val["y"], test["y"], full_results["yhat"], full_results["lower"], full_results["upper"]]:
        y_candidates.extend(series.replace([np.inf, -np.inf], np.nan).dropna().tolist())
    if not y_candidates:
        st.error("ARIMA produced no finite values to plot.")
        return False
    y_min, y_max = float(np.min(y_candidates)), float(np.max(y_candidates))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        st.error("ARIMA produced degenerate scale for plotting.")
        return False
    y_pad = (y_max - y_min) * 0.05
    y_min -= y_pad
    y_max += y_pad

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train["ds"], train["y"], label="Train", color="gray")
    ax.plot(val["ds"], val["y"], label="Validation", color="orange")
    ax.plot(test["ds"], test["y"], label="Test", color="green")

    ax.plot(full_results["ds"], full_results["yhat"], label="Forecast (ARIMA)", color="blue", linewidth=2)
    ax.fill_between(full_results["ds"], full_results["lower"], full_results["upper"], color="#92B6FF", alpha=0.25)

    ax.set_title(f"{ticker} ARIMA Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    ax.legend()
    ax.set_ylim(y_min, y_max)
    fig.autofmt_xdate()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8, prune=None))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, prune=None))
    try:
        plt.tight_layout()
    except Exception:
        # If tight_layout fails (e.g., due to bad ticks), skip so the app doesn't crash
        pass
    try:
        st.pyplot(fig)
    except Exception as exc:
        st.error(f"Plotting failed: {exc}")
        return False
    st.dataframe(full_results, use_container_width=True)
    return True


def run_lstm_forecast(hist_df: pd.DataFrame, ticker: str, forecast_horizon: int):
    try:
        import tensorflow as tf
        keras = tf.keras
    except Exception as exc:
        st.error(f"Failed to import TensorFlow/Keras: {exc}")
        return False


    data = hist_df[hist_df["Ticker"] == ticker][["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    data = data.sort_values("ds").dropna().reset_index(drop=True)

    if len(data) < 40:
        st.warning("Not enough data to run LSTM (need at least 40 rows).")
        return False

    train_size = int(len(data) * 0.8)
    val_size = int(len(data) * 0.1)
    if train_size <= 5 or val_size == 0:
        st.warning("Not enough data to create train/validation/test splits.")
        return False

    seq_len = 5

    def make_sequences(df):
        X, y, ds_list = [], [], []
        for i in range(seq_len, len(df)):
            window = df.iloc[i - seq_len : i]["y"].values
            target = df.iloc[i]["y"]
            X.append(window)
            y.append(target)
            ds_list.append(df.iloc[i]["ds"])
        return np.array(X), np.array(y), ds_list

    X_all, y_all, ds_all = make_sequences(data)

    train_idx = train_size - seq_len
    val_idx = train_size + val_size - seq_len

    X_train, y_train = X_all[:train_idx], y_all[:train_idx]
    X_val, y_val = X_all[train_idx:val_idx], y_all[train_idx:val_idx]
    X_test, y_test = X_all[val_idx:], y_all[val_idx:]
    ds_train = ds_all[:train_idx]
    ds_val = ds_all[train_idx:val_idx]
    ds_test = ds_all[val_idx:]

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    try:
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(seq_len, 1)),
                keras.layers.LSTM(32, return_sequences=False),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            X_train,
            y_train,
            epochs=30,
            batch_size=16,
            validation_data=(X_val, y_val),
            verbose=0,
        )
    except Exception as exc:
        st.error(f"Failed to train LSTM: {exc}")
        return False

    def predict_on_sequences(X_seq):
        return model.predict(X_seq, verbose=0).flatten()

    val_preds = predict_on_sequences(X_val) if len(X_val) else np.array([])
    test_preds = predict_on_sequences(X_test) if len(X_test) else np.array([])

    results_hist = pd.DataFrame(
        {
            "ds": ds_val + ds_test,
            "y": list(y_val) + list(y_test),
            "yhat": list(val_preds) + list(test_preds),
        }
    )

    # Future recursive forecast
    last_date = data["ds"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon)
    future_df = pd.DataFrame({"ds": future_dates})

    window = data["y"].values[-seq_len:].tolist()
    future_preds = []
    for _ in range(forecast_horizon):
        X_input = np.array(window[-seq_len:], dtype=float).reshape(1, seq_len, 1)
        yhat = float(model.predict(X_input, verbose=0).flatten()[0])
        future_preds.append(yhat)
        window.append(yhat)

    future_df["yhat"] = future_preds
    future_df["y"] = np.nan

    base_std = float(np.std(val_preds - y_val)) if len(y_val) > 1 else 0.0
    z_outer = 2.58
    horizon_idx = np.arange(len(results_hist) + len(future_df))
    widen = 1 + (horizon_idx / max(len(horizon_idx) - 1, 1)) * 1.5

    full_results = pd.concat([results_hist, future_df], ignore_index=True)
    full_results = full_results.sort_values("ds").reset_index(drop=True)
    full_results["lower"] = full_results["yhat"] - z_outer * base_std * widen
    full_results["upper"] = full_results["yhat"] + z_outer * base_std * widen

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ds_train, y_train, label="Train", color="gray")
    ax.plot(ds_val, y_val, label="Validation", color="orange")
    ax.plot(ds_test, y_test, label="Test", color="green")
    ax.plot(full_results["ds"], full_results["yhat"], label="Forecast (LSTM)", color="blue", linewidth=2)
    if base_std > 0:
        ax.fill_between(
            full_results["ds"],
            full_results["lower"],
            full_results["upper"],
            color="#92B6FF",
            alpha=0.25,
        )

    ax.set_title(f"{ticker} LSTM Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig)
    st.dataframe(full_results, use_container_width=True)
    return True

from neuralprophet import NeuralProphet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def run_neuralprophet_forecast(hist_df: pd.DataFrame, ticker: str, forecast_horizon: int = 30):
    # Prepare data
    df = (
    hist_df[hist_df["Ticker"] == ticker][["Date", "Close"]]
    .rename(columns={"Date": "ds", "Close": "y"})
    .sort_values("ds")
    .drop_duplicates(subset=["ds"], keep="last")    # ← FIX
    .dropna()
    .reset_index(drop=True)
    )

    if len(df) < 40:
        st.warning("Not enough data to run NeuralProphet (need at least 40 rows).")
        return False

    # Initialize model
    model = NeuralProphet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        learning_rate=0.01,
        epochs=50,
    )

    # Fit model
    metrics = model.fit(df, freq="D")

    # Make future dataframe
    future = model.make_future_dataframe(df, periods=forecast_horizon)
    forecast = model.predict(future)

    # Plot result
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["ds"], df["y"], label="Actual", color="black")
    ax.plot(forecast["ds"], forecast["yhat1"], label="Forecast", color="blue", linewidth=2)

    # Confidence-like bands (simulated using residual std)
    residuals = df["y"] - model.predict(df)["yhat1"]
    std = residuals.std()
    upper = forecast["yhat1"] + 1.96 * std
    lower = forecast["yhat1"] - 1.96 * std

    ax.fill_between(forecast["ds"], lower, upper, color="blue", alpha=0.2, label="Uncertainty")

    ax.set_title(f"NeuralProphet Forecast for {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    fig.autofmt_xdate()

    st.pyplot(fig)
    st.dataframe(forecast[["ds", "yhat1"]], use_container_width=True)

    return True

def render_forecast():
    st.title("🔮 Forecast")
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL"]
    ticker = st.selectbox("Choose ticker:", tickers, index=0)

    if st.button("Generate Forecast"):
        project_root = Path(__file__).resolve().parents[1]
        csv_path = project_root / "dataset" / "historical_stocks_clean.csv"
        if not csv_path.exists():
            st.error(f"Historical CSV not found at {csv_path}")
            return

        hist_df = load_historical_data(csv_path)
        df = (
            hist_df[hist_df["Ticker"] == ticker][["Date", "Close"]]
            .rename(columns={"Date": "ds", "Close": "y"})
            .sort_values("ds")
        )
        if df.empty:
            st.info(f"No data available for {ticker}.")
            return

        st.subheader(f"{ticker} historical close")
        st.line_chart(df.set_index("ds")["y"])

        returns_path = project_root / "dataset" / "returns.csv"
        if not returns_path.exists():
            st.warning(f"Returns file not found at {returns_path}")
            return
        try:
            returns_df = load_returns_data(returns_path)
            ticker_returns = returns_df[returns_df.get("ticker") == ticker]
        except Exception as exc:
            st.error(f"Failed to load returns: {exc}")
            return

        if ticker_returns.empty:
            st.info(f"No return forecasts available for {ticker}.")
            return

        latest = ticker_returns.sort_values("date").iloc[-1]
        fields = {k: latest.get(k, "") for k in [
            "date", "open", "high", "low", "close", "adj_close", "volume",
            "meta_pred_1d", "meta_pred_5d", "confidence_1d", "confidence_5d"
        ]}
        date_str = fields["date"].date().isoformat() if hasattr(fields["date"], "date") else str(fields["date"])

        def fmt_val(val, dec=6):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "—"
            if isinstance(val, (int, float, np.floating)):
                return f"{val:.{dec}f}"
            return str(val)

        card_lines = [
            f'<div style="border: 1px solid #2d3748; border-radius: 12px; padding: 16px; margin-top: 12px; background: linear-gradient(145deg, #1e293b, #0f172a); box-shadow: 0 4px 12px rgba(0,0,0,0.3); color: #e2e8f0;">',
            f'  <div style="font-size: 18px; font-weight: 700; margin-bottom: 10px; color:#f1f5f9;">{ticker} · Return Snapshot <span style="font-size:14px; color:#94a3b8;">({date_str})</span></div>',
            f'  <div style="font-size: 13px; color:#cbd5e1; margin-bottom: 14px; line-height: 1.4;">',
            f'    O/H/L/C: <span style="color:#38bdf8;">{fields["open"]}</span>, <span style="color:#38bdf8;">{fields["high"]}</span>, <span style="color:#38bdf8;">{fields["low"]}</span>, <span style="color:#38bdf8;">{fields["close"]}</span> · Adj Close: <span style="color:#cbd5e1;">{fields["adj_close"]}</span> · Volume: <span style="color:#cbd5e1;">{fields["volume"]}</span>',
            f'  </div>',
            f'  <div style="display:flex; gap:14px; flex-wrap:wrap;">',
            f'    <div style="background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.25); padding: 12px; border-radius: 10px; min-width: 160px; flex: 1;">',
            f'      <div style="font-size:12px; color:#bae6fd;">1-day forecast</div>',
            f'      <div style="font-size:20px; font-weight:700; margin:4px 0; color:#38bdf8;">{fmt_val(fields["meta_pred_1d"])}</div>',
            f'      <div style="font-size:12px; color:#93c5fd;">Confidence: {fmt_val(fields["confidence_1d"], dec=3)}</div>',
            f'    </div>',
            f'    <div style="background: rgba(139,92,246,0.15); border: 1px solid rgba(139,92,246,0.25); padding: 12px; border-radius: 10px; min-width: 160px; flex: 1;">',
            f'      <div style="font-size:12px; color:#ddd6fe;">5-day forecast</div>',
            f'      <div style="font-size:20px; font-weight:700; margin:4px 0; color:#a78bfa;">{fmt_val(fields["meta_pred_5d"])}</div>',
            f'      <div style="font-size:12px; color:#c4b5fd;">Confidence: {fmt_val(fields["confidence_5d"], dec=3)}</div>',
            f'    </div>',
            f'  </div>',
            f'</div>',
        ]
        card_html = "\n".join(card_lines)
        st.markdown(card_html, unsafe_allow_html=True)



if __name__ == "__main__":
    render_forecast()
