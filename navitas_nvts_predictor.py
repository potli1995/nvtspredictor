import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score


st.set_page_config(page_title="NVTS Daily Predictor", layout="wide")
st.title("NVTS Daily Predictor")
st.caption("Educational research tool only. Not financial advice. NVTS can be extremely volatile.")

DEFAULT_TICKER = "NVTS"

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value=DEFAULT_TICKER).upper().strip()
    period = st.selectbox("Training history", ["1y", "2y", "5y", "max"], index=2)
    buy_threshold = st.slider("BUY confidence threshold", 0.50, 0.80, 0.57, 0.01)
    strong_buy_threshold = st.slider("STRONG BUY threshold", 0.55, 0.90, 0.65, 0.01)

def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=600)
def load_daily(symbol, period):
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

@st.cache_data(ttl=300)
def load_intraday(symbol):
    df = yf.download(symbol, period="5d", interval="5m", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def make_features(df):
    data = df.copy()
    data["return_1d"] = data["Close"].pct_change()
    data["return_2d"] = data["Close"].pct_change(2)
    data["return_3d"] = data["Close"].pct_change(3)
    data["return_5d"] = data["Close"].pct_change(5)
    data["return_10d"] = data["Close"].pct_change(10)

    data["ma_5"] = data["Close"].rolling(5).mean()
    data["ma_10"] = data["Close"].rolling(10).mean()
    data["ma_20"] = data["Close"].rolling(20).mean()
    data["ma_50"] = data["Close"].rolling(50).mean()

    data["ma5_ma20"] = data["ma_5"] / data["ma_20"] - 1
    data["ma10_ma50"] = data["ma_10"] / data["ma_50"] - 1
    data["close_ma20"] = data["Close"] / data["ma_20"] - 1

    data["volatility_5"] = data["return_1d"].rolling(5).std()
    data["volatility_10"] = data["return_1d"].rolling(10).std()
    data["volatility_20"] = data["return_1d"].rolling(20).std()

    data["volume_change"] = data["Volume"].pct_change()
    data["volume_ratio_20"] = data["Volume"] / data["Volume"].rolling(20).mean()
    data["rsi_14"] = rsi(data["Close"], 14)

    data["range_pct"] = (data["High"] - data["Low"]) / data["Close"]
    data["gap_pct"] = data["Open"] / data["Close"].shift(1) - 1

    data["target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    return data.replace([np.inf, -np.inf], np.nan).dropna()

def train_models(data, features):
    split = int(len(data) * 0.80)
    train = data.iloc[:split]
    test = data.iloc[split:]

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42
    )
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.04,
        max_depth=2,
        random_state=42
    )

    rf.fit(train[features], train["target"])
    gb.fit(train[features], train["target"])

    probs = (
        rf.predict_proba(test[features])[:, 1] +
        gb.predict_proba(test[features])[:, 1]
    ) / 2
    preds = (probs >= 0.50).astype(int)

    return rf, gb, test, probs, preds

def signal_from_probability(prob, buy_threshold, strong_buy_threshold):
    if prob >= strong_buy_threshold:
        return "STRONG BUY / HIGH UP-BIAS"
    if prob >= buy_threshold:
        return "BUY / UP-BIAS"
    if prob >= 0.48:
        return "NEUTRAL / WAIT"
    return "BEARISH / AVOID LONG"

def get_levels(df):
    recent = df.tail(20)
    latest_close = float(df["Close"].iloc[-1])
    support_5 = float(df["Low"].tail(5).min())
    support_20 = float(recent["Low"].min())
    resistance_5 = float(df["High"].tail(5).max())
    resistance_20 = float(recent["High"].max())
    atr_14 = float((df["High"] - df["Low"]).rolling(14).mean().iloc[-1])
    stop_example = latest_close - atr_14
    target_example = latest_close + atr_14
    return latest_close, support_5, support_20, resistance_5, resistance_20, atr_14, stop_example, target_example

try:
    raw = load_daily(ticker, period)

    if raw.empty or len(raw) < 120:
        st.error("Not enough daily data found. Try a longer period or check the ticker.")
        st.stop()

    data = make_features(raw)

    features = [
        "return_1d", "return_2d", "return_3d", "return_5d", "return_10d",
        "ma5_ma20", "ma10_ma50", "close_ma20",
        "volatility_5", "volatility_10", "volatility_20",
        "volume_change", "volume_ratio_20",
        "rsi_14", "range_pct", "gap_pct"
    ]

    rf, gb, test, probs, preds = train_models(data, features)

    latest = data.iloc[[-1]]
    latest_prob = float((
        rf.predict_proba(latest[features])[:, 1][0] +
        gb.predict_proba(latest[features])[:, 1][0]
    ) / 2)

    signal = signal_from_probability(latest_prob, buy_threshold, strong_buy_threshold)
    acc = accuracy_score(test["target"], preds)
    precision_when_up = precision_score(test["target"], preds, zero_division=0)

    latest_close, support_5, support_20, resistance_5, resistance_20, atr_14, stop_example, target_example = get_levels(raw)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest close", f"${latest_close:,.2f}")
    c2.metric("Next-day up probability", f"{latest_prob:.1%}")
    c3.metric("Backtest accuracy", f"{acc:.1%}")
    c4.metric("Precision when bullish", f"{precision_when_up:.1%}")

    st.subheader(f"Today’s signal: {signal}")

    if signal.startswith("STRONG") or signal.startswith("BUY"):
        st.success("Bullish setup. Consider only if price confirms with volume and does not lose key support.")
    elif signal.startswith("NEUTRAL"):
        st.warning("No clean edge. Best action may be to wait for breakout or pullback confirmation.")
    else:
        st.error("Weak setup. Avoid chasing unless the chart strongly reverses with volume.")

    st.subheader("Key levels")
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("5-day support", f"${support_5:,.2f}")
    l2.metric("20-day support", f"${support_20:,.2f}")
    l3.metric("5-day resistance", f"${resistance_5:,.2f}")
    l4.metric("20-day resistance", f"${resistance_20:,.2f}")

    st.write(f"**ATR-style daily move estimate:** about **${atr_14:,.2f}** from the latest close.")
    st.write(f"Example risk zone: stop near **${stop_example:,.2f}**, upside target near **${target_example:,.2f}**. Adjust manually using live price action.")

    st.subheader("Daily close chart")
    st.line_chart(raw["Close"])

    try:
        intraday = load_intraday(ticker)
        if not intraday.empty:
            st.subheader("Recent 5-minute intraday chart")
            st.line_chart(intraday["Close"])
    except Exception:
        pass

    st.subheader("Today’s trading checklist")
    st.markdown(f""" - Prefer long only if price holds above **5-day support (${support_5:,.2f})**. - Breakout confirmation improves above **5-day resistance (${resistance_5:,.2f})** with strong volume. - Avoid chasing if price is extended far above the open without pullbacks. - Cut risk fast if the stock loses support with heavy selling volume. - This tool predicts probability, not certainty. """)

    result_df = test.copy()
    result_df["up_probability"] = probs
    result_df["prediction"] = np.where(result_df["up_probability"] >= 0.50, "UP", "DOWN")

    st.subheader("Recent model backtest")
    st.dataframe(
        result_df[["Close", "target", "up_probability", "prediction"]].tail(40),
        use_container_width=True
    )

    st.subheader("Feature importance")
    importance = pd.DataFrame({
        "feature": features,
        "random_forest_importance": rf.feature_importances_
    }).sort_values("random_forest_importance", ascending=False)
    st.bar_chart(importance.set_index("feature"))
    except Exception as e:
    st.error(f"Error: {e}")