"""
main.py
────────
StockVision AI — Main Streamlit entry point.
Run: streamlit run main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import os
import sys
import logging

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from utils.data_handler import (
    fetch_stock_data, get_stock_info, add_technical_indicators,
    prepare_lstm_data, inverse_transform_predictions,
    generate_trading_signals, calculate_portfolio_metrics,
    POPULAR_STOCKS, TIMEFRAME_MAP,
)
from utils.charts import (
    candlestick_chart, prediction_chart, rsi_chart, macd_chart,
    bollinger_chart, loss_curves, portfolio_donut, signal_chart,
    correlation_heatmap,
)
from utils.sentiment import fetch_news, analyze_sentiment, aggregate_sentiment
from utils.ai_insights import generate_insights
from models.lstm_model import (
    build_lstm_model, build_gru_model, train_model,
    evaluate_model, forecast_future, save_model,
)

logging.basicConfig(level=logging.WARNING)
os.makedirs(os.path.join(ROOT, "data", "saved_models"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "data", "exports"), exist_ok=True)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockVision AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {
  --bg:       #0D0F14;
  --bg2:      #151821;
  --bg3:      #1A1F2E;
  --border:   #1E2330;
  --accent:   #00D4FF;
  --green:    #00E676;
  --red:      #FF4444;
  --yellow:   #FFD600;
  --purple:   #A855F7;
  --text:     #E2E8F0;
  --muted:    #64748B;
  --font:     'IBM Plex Mono', monospace;
  --font2:    'Space Grotesk', sans-serif;
}

/* ── Root overrides ── */
html, body, .stApp { background-color: var(--bg) !important; color: var(--text); }
.stApp { font-family: var(--font2); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stSlider { margin-bottom: 0.5rem; }

/* ── Metrics / KPI Cards ── */
[data-testid="stMetric"] {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.7rem !important; letter-spacing: 0.08em; text-transform: uppercase; font-family: var(--font); }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: var(--font); font-size: 1.4rem !important; }
[data-testid="stMetricDelta"] { font-family: var(--font); font-size: 0.8rem !important; }

/* ── Expander ── */
.streamlit-expander {
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #00D4FF22, #A855F722) !important;
  border: 1px solid var(--accent) !important;
  color: var(--accent) !important;
  font-family: var(--font) !important;
  letter-spacing: 0.05em;
  border-radius: 6px !important;
  transition: all 0.2s;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #00D4FF44, #A855F744) !important;
  box-shadow: 0 0 14px #00D4FF44;
}

/* ── Info / Warning boxes ── */
.stAlert { border-radius: 8px !important; font-family: var(--font); font-size: 0.85rem; }

/* ── Section header utility ── */
.sv-header {
  font-family: var(--font2);
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--accent);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.4rem;
  margin: 1.2rem 0 0.8rem;
}

/* ── Signal badges ── */
.signal-buy  { color: #00E676; font-weight: 700; font-family: var(--font); }
.signal-sell { color: #FF4444; font-weight: 700; font-family: var(--font); }
.signal-hold { color: #FFD600; font-weight: 700; font-family: var(--font); }

/* ── Logo ── */
.logo-text {
  font-family: var(--font);
  font-size: 1.35rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  color: var(--accent);
}
.logo-sub {
  font-family: var(--font);
  font-size: 0.65rem;
  color: var(--muted);
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

/* ── Insight card ── */
.insight-card {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 6px;
  padding: 0.7rem 1rem;
  margin-bottom: 0.6rem;
  font-size: 0.88rem;
  line-height: 1.6;
}

/* ── DataFrames ── */
.stDataFrame { background: var(--bg3) !important; border: 1px solid var(--border) !important; border-radius: 8px; }
thead { background: var(--border) !important; }

/* ── Progress / spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── Session State Init ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "df_raw":       None,
        "df_ind":       None,
        "df_signals":   None,
        "stock_info":   {},
        "ticker":       "AAPL",
        "period":       "1 Year",
        "model":        None,
        "history":      None,
        "metrics":      {},
        "preds_test":   None,
        "actual_test":  None,
        "future_preds": None,
        "scaler_t":     None,
        "train_size":   None,
        "sentiment_df": None,
        "sentiment_agg":{},
        "portfolio":    [],
        "page":         "Home",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════════════════════
# ── Sidebar ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="logo-text">📈 StockVision AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">LSTM Prediction Platform</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Navigation
    pages = ["Home", "Prediction Dashboard", "Technical Analysis",
             "Portfolio Analytics", "AI Insights", "About"]
    page = st.radio("Navigate", pages, index=pages.index(st.session_state.page),
                    label_visibility="collapsed")
    st.session_state.page = page
    st.markdown("---")

    # Stock selector
    st.markdown('<div class="sv-header" style="margin-top:0">Stock Settings</div>',
                unsafe_allow_html=True)

    market = st.selectbox("Market", list(POPULAR_STOCKS.keys()), index=0)
    preset = st.selectbox("Quick Select", ["Custom"] + POPULAR_STOCKS[market])
    if preset != "Custom":
        ticker_input = preset
    else:
        ticker_input = st.text_input("Ticker Symbol", value=st.session_state.ticker,
                                      max_chars=12, placeholder="e.g. AAPL").upper().strip()

    timeframe = st.selectbox("Timeframe", list(TIMEFRAME_MAP.keys()), index=3)
    period_str, interval = TIMEFRAME_MAP[timeframe]
    st.markdown("---")

    # LSTM settings
    st.markdown('<div class="sv-header" style="margin-top:0">LSTM Config</div>',
                unsafe_allow_html=True)
    window     = st.slider("Lookback Window", 20, 120, 60, 10)
    epochs     = st.slider("Max Epochs",      20, 200, 80, 10)
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    dropout    = st.slider("Dropout Rate", 0.1, 0.5, 0.2, 0.05)
    lstm_units = st.multiselect("LSTM Layers (units)",
                                [32, 64, 128, 256], default=[128, 64, 32])
    future_days= st.slider("Forecast Horizon (days)", 5, 90, 30, 5)
    compare_gru= st.checkbox("Compare with GRU", value=False)
    st.markdown("---")

    fetch_btn = st.button("🔄 Fetch Data", use_container_width=True)
    train_btn = st.button("🧠 Train LSTM", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── Data Fetch ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if fetch_btn or (st.session_state.df_raw is None and ticker_input):
    with st.spinner(f"Fetching {ticker_input} data…"):
        try:
            st.session_state.ticker = ticker_input
            df_raw = fetch_stock_data(ticker_input, period=period_str, interval=interval)
            df_ind = add_technical_indicators(df_raw)
            df_sig = generate_trading_signals(df_ind)
            info   = get_stock_info(ticker_input)

            st.session_state.df_raw     = df_raw
            st.session_state.df_ind     = df_ind
            st.session_state.df_signals = df_sig
            st.session_state.stock_info = info
            st.session_state.period     = timeframe

            # Reset model state on new fetch
            st.session_state.model  = None
            st.session_state.metrics= {}
            st.sidebar.success(f"✓ {len(df_raw)} rows loaded")
        except Exception as e:
            st.sidebar.error(f"Fetch error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ── LSTM Training ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if train_btn:
    if st.session_state.df_ind is None:
        st.sidebar.warning("Fetch data first.")
    else:
        with st.spinner("Training LSTM… (this may take 1–3 minutes)"):
            try:
                df_ind = st.session_state.df_ind
                feature_cols = [
                    "Open", "High", "Low", "Volume",
                    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
                    "RSI_14", "MACD", "MACD_Signal",
                    "BB_Upper", "BB_Lower", "BB_Width",
                    "ATR_14", "Daily_Return",
                ]
                feature_cols = [c for c in feature_cols if c in df_ind.columns]

                X_tr, y_tr, X_te, y_te, sc_f, sc_t, train_size = prepare_lstm_data(
                    df_ind, feature_cols=feature_cols,
                    window=window, test_ratio=0.2,
                )

                val_split = int(len(X_tr) * 0.85)
                X_val, y_val = X_tr[val_split:], y_tr[val_split:]
                X_tr2, y_tr2 = X_tr[:val_split], y_tr[:val_split]

                model = build_lstm_model(
                    input_shape=(X_tr.shape[1], X_tr.shape[2]),
                    units=lstm_units or [128, 64, 32],
                    dropout=dropout,
                )
                history, model = train_model(
                    model, X_tr2, y_tr2, X_val, y_val,
                    epochs=epochs, batch_size=batch_size,
                    ticker=ticker_input,
                )
                metrics = evaluate_model(model, X_te, y_te, sc_t)

                # Future forecast
                last_seq = X_te[-1]
                future   = forecast_future(model, last_seq, sc_t, n_steps=future_days)

                st.session_state.model       = model
                st.session_state.history     = history
                st.session_state.metrics     = metrics
                st.session_state.preds_test  = metrics["predictions"]
                st.session_state.actual_test = metrics["actual"]
                st.session_state.future_preds= future
                st.session_state.scaler_t    = sc_t
                st.session_state.train_size  = train_size

                # GRU comparison
                if compare_gru:
                    gru = build_gru_model(input_shape=(X_tr.shape[1], X_tr.shape[2]))
                    _, gru = train_model(gru, X_tr2, y_tr2, X_val, y_val,
                                         epochs=epochs, batch_size=batch_size, ticker="GRU")
                    gru_m = evaluate_model(gru, X_te, y_te, sc_t)
                    st.session_state.gru_metrics = gru_m
                else:
                    st.session_state.gru_metrics = None

                st.sidebar.success(f"✓ Accuracy: {metrics['Accuracy']:.1f}%")

            except Exception as e:
                st.sidebar.error(f"Training error: {e}")
                import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# ── Helper: KPI row ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def kpi_row():
    df  = st.session_state.df_raw
    info= st.session_state.stock_info
    if df is None:
        return
    last  = df["Close"].iloc[-1]
    prev  = df["Close"].iloc[-2] if len(df) > 1 else last
    chg   = last - prev
    chg_p = chg / prev * 100 if prev else 0
    vol   = df["Volume"].iloc[-1] if "Volume" in df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Price",    f"{last:,.2f}", f"{chg:+.2f} ({chg_p:+.2f}%)")
    c2.metric("📊 Volume",   f"{vol:,.0f}")
    c3.metric("52W High",    f"{info.get('52w_high','—')}")
    c4.metric("52W Low",     f"{info.get('52w_low','—')}")
    mktcap = info.get("market_cap", 0)
    c5.metric("Market Cap",  f"${mktcap/1e9:.2f}B" if mktcap else "—")


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: HOME ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    info = st.session_state.stock_info
    tick = st.session_state.ticker

    st.markdown(f"""
    <h1 style='font-family: IBM Plex Mono; color:#00D4FF; margin-bottom:0'>
    StockVision <span style='color:#A855F7'>AI</span>
    </h1>
    <p style='color:#64748B; font-family: IBM Plex Mono; font-size:0.8rem; letter-spacing:0.1em; margin-top:4px'>
    LSTM DEEP LEARNING · TRADING ANALYTICS · PORTFOLIO INTELLIGENCE
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.df_raw is not None:
        kpi_row()
        st.markdown(f"<div class='sv-header'>{info.get('name', tick)} — {info.get('sector','')}</div>",
                    unsafe_allow_html=True)
        if info.get("description"):
            st.caption(info["description"][:420] + "…")

        st.plotly_chart(
            candlestick_chart(st.session_state.df_raw, tick),
            use_container_width=True,
        )

        # Latest signal
        df_s = st.session_state.df_signals
        if df_s is not None:
            sig = df_s["Signal_Label"].iloc[-1]
            color = "signal-buy" if sig == "BUY" else ("signal-sell" if sig == "SELL" else "signal-hold")
            st.markdown(
                f"<p style='font-family:IBM Plex Mono'>Latest AI Signal: "
                f"<span class='{color}'>{sig}</span></p>",
                unsafe_allow_html=True,
            )
    else:
        st.info("👈  Select a stock and click **Fetch Data** to begin.")
        st.markdown("""
        ### What StockVision AI can do
        | Feature | Description |
        |---|---|
        | 🧠 LSTM Prediction | Multi-layer deep learning price forecast |
        | 📊 Technical Analysis | RSI, MACD, Bollinger Bands, SMA, EMA |
        | 🤖 AI Trading Signals | Buy / Hold / Sell recommendations |
        | 💼 Portfolio Tracker | Real-time P&L monitoring |
        | 📰 Sentiment Analysis | NLP-scored news sentiment |
        | 🔮 AI Insights | Plain-English market commentary |
        """)


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: PREDICTION DASHBOARD ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction Dashboard":
    st.markdown("<h2 style='font-family:IBM Plex Mono;color:#00D4FF'>🧠 LSTM Prediction Dashboard</h2>",
                unsafe_allow_html=True)

    if st.session_state.df_raw is None:
        st.info("Fetch stock data first.")
        st.stop()

    kpi_row()

    if st.session_state.model is None:
        st.warning("Train the LSTM model using the sidebar to see predictions.")
    else:
        metrics = st.session_state.metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("RMSE",     f"{metrics['RMSE']:.4f}")
        m2.metric("MAE",      f"{metrics['MAE']:.4f}")
        m3.metric("MAPE",     f"{metrics['MAPE']:.2f}%")
        m4.metric("R²",       f"{metrics['R2']:.4f}")
        m5.metric("Accuracy", f"{metrics['Accuracy']:.1f}%",
                  delta="Good" if metrics["Accuracy"] >= 85 else "Retrain?")

        st.markdown("---")

        df_ind = st.session_state.df_ind
        train_size = st.session_state.train_size
        preds  = st.session_state.preds_test
        actual = st.session_state.actual_test
        future = st.session_state.future_preds

        # Build date arrays
        dates_all   = df_ind.index.tolist()
        dates_train = dates_all[:train_size + window]
        dates_test  = dates_all[train_size + window: train_size + window + len(actual)]
        last_date   = dates_test[-1] if dates_test else datetime.now()
        dates_future= [last_date + timedelta(days=i+1) for i in range(len(future))]

        actual_train = df_ind["Close"].values[:train_size + window]

        # Get window from config (approximate)
        window_size = len(dates_all) - train_size - len(actual)

        fig = prediction_chart(
            dates_train, actual_train,
            dates_test, actual, preds,
            dates_future, future,
            st.session_state.ticker,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Loss curves
        with st.expander("📉 Training Loss Curves"):
            st.plotly_chart(loss_curves(st.session_state.history),
                            use_container_width=True)

        # GRU comparison
        gru_m = st.session_state.get("gru_metrics")
        if gru_m:
            st.markdown("<div class='sv-header'>Model Comparison</div>",
                        unsafe_allow_html=True)
            comp = pd.DataFrame([
                {"Model": "LSTM", **{k: v for k, v in metrics.items()
                                     if k not in ("predictions","actual")}},
                {"Model": "GRU",  **{k: v for k, v in gru_m.items()
                                     if k not in ("predictions","actual")}},
            ])
            st.dataframe(comp.set_index("Model"), use_container_width=True)

        # Export
        st.markdown("<div class='sv-header'>Export</div>", unsafe_allow_html=True)
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            test_df = pd.DataFrame({
                "Date":      dates_test,
                "Actual":    actual,
                "Predicted": preds,
            })
            csv = test_df.to_csv(index=False)
            st.download_button("⬇ Download Test Predictions CSV", csv,
                               file_name=f"{st.session_state.ticker}_predictions.csv",
                               mime="text/csv")
        with col_exp2:
            fut_df = pd.DataFrame({"Date": dates_future, "Forecast": future})
            csv2 = fut_df.to_csv(index=False)
            st.download_button("⬇ Download Forecast CSV", csv2,
                               file_name=f"{st.session_state.ticker}_forecast.csv",
                               mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: TECHNICAL ANALYSIS ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Technical Analysis":
    st.markdown("<h2 style='font-family:IBM Plex Mono;color:#00D4FF'>📊 Technical Analysis</h2>",
                unsafe_allow_html=True)

    if st.session_state.df_ind is None:
        st.info("Fetch data first.")
        st.stop()

    df  = st.session_state.df_ind
    sig = st.session_state.df_signals
    tick= st.session_state.ticker

    kpi_row()
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Candlestick", "RSI", "MACD", "Bollinger Bands", "Signals"
    ])

    with tab1:
        st.plotly_chart(candlestick_chart(df, tick), use_container_width=True)

    with tab2:
        st.plotly_chart(rsi_chart(df, tick), use_container_width=True)
        with st.expander("RSI Interpretation"):
            st.markdown("""
            - **RSI > 70**: Overbought — potential sell signal
            - **RSI < 30**: Oversold — potential buy signal
            - **RSI 40–60**: Neutral zone
            """)

    with tab3:
        st.plotly_chart(macd_chart(df, tick), use_container_width=True)
        with st.expander("MACD Interpretation"):
            st.markdown("""
            - **MACD crosses above Signal**: Bullish crossover → potential BUY
            - **MACD crosses below Signal**: Bearish crossover → potential SELL
            - **Histogram growing**: Momentum increasing
            """)

    with tab4:
        st.plotly_chart(bollinger_chart(df, tick), use_container_width=True)

    with tab5:
        st.plotly_chart(signal_chart(sig, tick), use_container_width=True)
        signal_counts = sig["Signal_Label"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 BUY Signals",  signal_counts.get("BUY", 0))
        c2.metric("🟡 HOLD Signals", signal_counts.get("HOLD", 0))
        c3.metric("🔴 SELL Signals", signal_counts.get("SELL", 0))

    st.markdown("<div class='sv-header'>Indicator Correlation</div>",
                unsafe_allow_html=True)
    st.plotly_chart(correlation_heatmap(df), use_container_width=True)

    st.markdown("<div class='sv-header'>Raw Indicator Data</div>",
                unsafe_allow_html=True)
    display_cols = ["Close", "SMA_20", "SMA_50", "EMA_12",
                    "RSI_14", "MACD", "BB_Upper", "BB_Lower", "ATR_14"]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[display_cols].tail(50).style.format("{:.3f}"),
        use_container_width=True, height=320,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: PORTFOLIO ANALYTICS ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Portfolio Analytics":
    st.markdown("<h2 style='font-family:IBM Plex Mono;color:#00D4FF'>💼 Portfolio Analytics</h2>",
                unsafe_allow_html=True)

    st.markdown("<div class='sv-header'>Add Holdings</div>", unsafe_allow_html=True)

    col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
    with col_a:
        pticker = st.text_input("Ticker", placeholder="AAPL", key="pt").upper().strip()
    with col_b:
        pshares = st.number_input("Shares", min_value=0.01, value=10.0, step=1.0, key="ps")
    with col_c:
        pcost   = st.number_input("Avg Cost ($)", min_value=0.01, value=150.0, step=0.5, key="pc")
    with col_d:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add"):
            if pticker:
                st.session_state.portfolio.append({
                    "ticker": pticker, "shares": pshares, "avg_cost": pcost
                })
                st.success(f"Added {pticker}")

    if st.session_state.portfolio:
        with st.spinner("Fetching live prices…"):
            result = calculate_portfolio_metrics(st.session_state.portfolio)

        hdf = result["holdings_df"]

        st.markdown("<div class='sv-header'>Portfolio Summary</div>", unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        pnl_color = "normal" if result["total_pnl"] >= 0 else "inverse"
        p1.metric("Total Invested",  f"${result['total_invested']:,.2f}")
        p2.metric("Current Value",   f"${result['total_current']:,.2f}")
        p3.metric("Total P&L",       f"${result['total_pnl']:,.2f}",
                  f"{result['total_pnl_pct']:+.2f}%",
                  delta_color=pnl_color)
        p4.metric("Holdings", len(st.session_state.portfolio))

        col_l, col_r = st.columns([1.2, 1])
        with col_l:
            st.dataframe(
                hdf.style
                   .format({"Invested": "${:,.2f}", "Current Value": "${:,.2f}",
                             "P&L": "${:,.2f}", "P&L %": "{:+.2f}%"})
                   .applymap(lambda v: "color:#00E676" if isinstance(v, (int, float)) and v > 0
                             else ("color:#FF4444" if isinstance(v, (int, float)) and v < 0 else ""),
                             subset=["P&L", "P&L %"]),
                use_container_width=True,
            )
        with col_r:
            st.plotly_chart(portfolio_donut(hdf), use_container_width=True)

        if st.button("🗑 Clear Portfolio"):
            st.session_state.portfolio = []
            st.rerun()
    else:
        st.info("Add holdings above to track your portfolio P&L.")


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: AI INSIGHTS ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif page == "AI Insights":
    st.markdown("<h2 style='font-family:IBM Plex Mono;color:#00D4FF'>🔮 AI Insights</h2>",
                unsafe_allow_html=True)

    if st.session_state.df_ind is None:
        st.info("Fetch data first.")
        st.stop()

    tick = st.session_state.ticker
    df   = st.session_state.df_ind
    metrics = st.session_state.metrics

    # Sentiment
    st.markdown("<div class='sv-header'>News Sentiment Analysis</div>",
                unsafe_allow_html=True)

    if st.button("📰 Fetch Latest News"):
        with st.spinner("Fetching and scoring news…"):
            articles  = fetch_news(tick, days_back=7, max_articles=15)
            sent_df   = analyze_sentiment(articles)
            sent_agg  = aggregate_sentiment(sent_df)
            st.session_state.sentiment_df  = sent_df
            st.session_state.sentiment_agg = sent_agg

    agg = st.session_state.sentiment_agg
    if agg:
        sa, sb, sc = st.columns(3)
        sa.metric("Sentiment Score", f"{agg['score']:.0f}/100")
        sb.metric("Overall Outlook",  agg["overall"])
        sc.metric("Articles Analysed", agg["positive"] + agg["negative"] + agg["neutral"])

        sent_df = st.session_state.sentiment_df
        if sent_df is not None and not sent_df.empty:
            for _, row in sent_df.iterrows():
                color = "#00E676" if row["label"] == "Positive" else (
                        "#FF4444" if row["label"] == "Negative" else "#FFD600")
                st.markdown(
                    f"""<div class='insight-card'>
                    <span style='color:{color};font-weight:700'>[{row['label']}]</span>
                    &nbsp;{row['title']}<br>
                    <span style='color:#64748B;font-size:0.75rem'>{row['source']} · Score: {row['score_pct']:.0f}/100</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # AI Insights
    st.markdown("<div class='sv-header'>Market Intelligence Report</div>",
                unsafe_allow_html=True)

    insights = generate_insights(df, metrics, agg)
    for ins in insights:
        st.markdown(f"<div class='insight-card'>{ins}</div>", unsafe_allow_html=True)

    # Forecast summary
    if st.session_state.future_preds is not None:
        future = st.session_state.future_preds
        direction = "📈 Upward" if future[-1] > future[0] else "📉 Downward"
        chg_pct = (future[-1] - future[0]) / future[0] * 100
        st.markdown(
            f"""<div class='insight-card'>
            LSTM <b>{future_days if 'future_days' in dir() else 30}-day Forecast</b>:
            {direction} trajectory. Projected change: <b>{chg_pct:+.2f}%</b>.
            Target range: <b>{min(future):.2f} – {max(future):.2f}</b>
            </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE: ABOUT ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown("<h2 style='font-family:IBM Plex Mono;color:#00D4FF'>ℹ About StockVision AI</h2>",
                unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:IBM Plex Mono; color:#E2E8F0; line-height:1.9'>

    <h3 style='color:#A855F7'>Architecture Overview</h3>

    StockVision AI is a production-grade stock prediction platform powered by
    multi-layer LSTM neural networks, real-time market data, and NLP sentiment analysis.

    <h4 style='color:#00D4FF; margin-top:1.5rem'>Tech Stack</h4>

    | Layer | Technology |
    |---|---|
    | Frontend | Streamlit + Custom CSS |
    | ML Model | TensorFlow / Keras LSTM |
    | Data | yfinance (real-time) |
    | Indicators | `ta` library |
    | Sentiment | TextBlob + NewsAPI |
    | Visualisation | Plotly (interactive) |

    <h4 style='color:#00D4FF; margin-top:1.5rem'>LSTM Architecture</h4>

    ```
    Input (window × features)
        └─ BiLSTM(128) → BatchNorm
        └─ LSTM(64)    → Dropout(0.2)
        └─ LSTM(32)    → BatchNorm
        └─ Dropout(0.2)
        └─ Dense(16, relu)
        └─ Dense(1)           ← price prediction
    ```

    <h4 style='color:#00D4FF; margin-top:1.5rem'>Key Features</h4>

    ✅ Real-time NSE / NASDAQ / Crypto data<br>
    ✅ 16-feature LSTM with sliding window<br>
    ✅ Multi-step future forecasting<br>
    ✅ 6 technical indicators (RSI, MACD, BB, SMA, EMA, ATR)<br>
    ✅ Rule-based AI trading signals<br>
    ✅ NLP news sentiment analysis<br>
    ✅ Portfolio P&L tracker<br>
    ✅ GRU comparison mode<br>
    ✅ CSV export<br>

    <h4 style='color:#00D4FF; margin-top:1.5rem'>Disclaimer</h4>

    <span style='color:#64748B; font-size:0.8rem'>
    This tool is for educational and research purposes only.
    It does not constitute financial advice. Past performance is not indicative of future results.
    Always consult a qualified financial advisor before making investment decisions.
    </span>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1E2330; margin-top:3rem'>
<p style='text-align:center; color:#64748B; font-family:IBM Plex Mono; font-size:0.7rem'>
StockVision AI · Built with TensorFlow, Streamlit & Plotly · For educational use only
</p>
""", unsafe_allow_html=True)
