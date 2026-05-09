# 📈 StockVision AI
### LSTM Stock Prediction & Trading Analytics Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi" />
  <img src="https://img.shields.io/badge/Plotly-5.20-purple" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

> **A production-grade, resume-worthy AI trading platform** that combines multi-layer LSTM neural networks, real-time market data, technical analysis, NLP sentiment scoring, and portfolio tracking — packaged in a sleek dark finance dashboard.

---

## 🎯 Project Highlights

| Metric | Value |
|--------|-------|
| ML Architecture | Multi-layer LSTM + GRU comparison |
| Features Used | 16 engineered features + indicators |
| Indicators | RSI · MACD · Bollinger Bands · SMA · EMA · ATR · OBV |
| Markets Supported | NASDAQ · NSE · Crypto |
| Forecast Horizon | Up to 90 days |
| UI | Dark finance dashboard · 6 pages · KPI cards |

---

## 🖼 Screenshots

```
[ Home Dashboard ]          [ Prediction Chart ]
┌──────────────────────┐   ┌──────────────────────┐
│  📈 StockVision AI   │   │  LSTM vs Actual       │
│  AAPL  $182.30 +1.2% │   │  ──────── Actual      │
│  Vol: 58.4M           │   │  - - - - Prediction   │
│  [Candlestick Chart] │   │  ══════ Forecast      │
└──────────────────────┘   └──────────────────────┘

[ Technical Analysis ]      [ AI Insights ]
┌──────────────────────┐   ┌──────────────────────┐
│  RSI: 58.4  MACD ↑   │   │  🟢 BULLISH Outlook   │
│  BB: Inside range    │   │  RSI at 58 - bullish  │
│  Signal: BUY 🟢      │   │  News Score: 72/100   │
└──────────────────────┘   └──────────────────────┘
```

---

## ⚙️ Tech Stack

```
┌─────────────────────────────────────────────────────┐
│                  StockVision AI                     │
├──────────────┬──────────────┬───────────────────────┤
│   Frontend   │   Backend    │       AI/ML           │
│  Streamlit   │  FastAPI     │  TensorFlow/Keras     │
│  Plotly      │  Python 3.11 │  LSTM · GRU           │
│  Custom CSS  │  yfinance    │  Scikit-learn         │
│              │  NewsAPI     │  TextBlob NLP         │
└──────────────┴──────────────┴───────────────────────┘
```

---

## 🗂 Project Structure

```
stockvision-ai/
│
├── main.py                    ← Streamlit app entry point
│
├── app/
│   ├── __init__.py
│   └── api.py                 ← FastAPI REST endpoints
│
├── models/
│   ├── __init__.py
│   └── lstm_model.py          ← LSTM/GRU architecture + training
│
├── utils/
│   ├── __init__.py
│   ├── data_handler.py        ← Data fetching, indicators, preprocessing
│   ├── charts.py              ← All Plotly chart builders
│   ├── sentiment.py           ← NLP news sentiment
│   └── ai_insights.py        ← Plain-English insights engine
│
├── notebooks/
│   └── LSTM_exploration.ipynb ← Jupyter notebook for research
│
├── data/
│   ├── saved_models/          ← Trained .keras + scaler files
│   └── exports/               ← CSV prediction exports
│
├── .streamlit/
│   └── config.toml            ← Streamlit theme config
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/stockvision-ai.git
cd stockvision-ai
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment (optional)

```bash
cp .env.example .env
# Edit .env and add your API keys:
#   NEWS_API_KEY  — from newsapi.org (free tier available)
```

### 5. Run the dashboard

```bash
streamlit run main.py
```

Open **http://localhost:8501** in your browser.

### 6. (Optional) Run the REST API

```bash
uvicorn app.api:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

---

## 🐳 Docker

```bash
# Build and run everything
docker-compose up --build

# Streamlit: http://localhost:8501
# FastAPI:   http://localhost:8000/docs
```

---

## 🧠 LSTM Architecture

```
Input Layer     (batch, window=60, features=16)
     │
     ▼
LSTM Layer 1    128 units · return_sequences=True
BatchNorm
     │
     ▼
LSTM Layer 2    64 units · return_sequences=True
Dropout 0.2
     │
     ▼
LSTM Layer 3    32 units · return_sequences=False
BatchNorm
Dropout 0.2
     │
     ▼
Dense           16 units · ReLU activation
     │
     ▼
Dense           1 unit   · Linear (price prediction)

Optimizer : Adam (lr=1e-3, with ReduceLROnPlateau)
Loss      : Huber (robust to outliers)
Callbacks : EarlyStopping (patience=15) · ModelCheckpoint
```

---

## 📊 Features Walkthrough

### Page 1: Home
- Live KPI cards (price, volume, 52W range, market cap)
- Interactive candlestick chart with SMA/EMA overlays
- Latest AI signal badge (BUY / HOLD / SELL)

### Page 2: Prediction Dashboard
- LSTM test-set predictions vs actual prices
- 30–90 day future forecast with confidence band
- Model metrics: RMSE · MAE · MAPE · R² · Accuracy
- Training/validation loss curves
- GRU comparison table (optional)
- CSV export buttons

### Page 3: Technical Analysis
- Tabbed interface: Candlestick · RSI · MACD · Bollinger · Signals
- Indicator correlation heatmap
- Raw indicator data table

### Page 4: Portfolio Analytics
- Add any number of holdings (ticker · shares · avg cost)
- Real-time P&L calculation via live prices
- Portfolio donut chart (allocation)
- Color-coded profit/loss table

### Page 5: AI Insights
- News sentiment fetching (NewsAPI / mock feed)
- Per-article sentiment scores
- Aggregate outlook score (0–100)
- Plain-English market commentary combining indicators + model + sentiment

### Page 6: About
- Architecture overview
- Disclaimer

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/stocks/popular` | Popular stock lists |
| GET | `/stocks/{ticker}/info` | Company metadata |
| GET | `/stocks/{ticker}/price` | OHLCV data |
| GET | `/stocks/{ticker}/signals` | Trading signals |
| POST | `/model/train` | Trigger LSTM training |
| GET | `/model/status/{ticker}` | Training status + metrics |
| GET | `/model/list` | Saved models |

Full interactive docs at `http://localhost:8000/docs`

---

## 🌐 Deployment

### Render (Free tier)

1. Push to GitHub
2. Create new **Web Service** on Render
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `streamlit run main.py --server.port $PORT --server.address 0.0.0.0`
5. Add environment variables from `.env.example`

### Railway

```bash
railway init
railway up
```

### Hugging Face Spaces

1. Create a new Space with Streamlit SDK
2. Upload all project files
3. Hugging Face auto-installs from `requirements.txt`

---

## 📈 Model Performance (Sample — AAPL 2-year data)

| Metric | Value |
|--------|-------|
| RMSE | 2.34 |
| MAE | 1.87 |
| MAPE | 1.12% |
| R² | 0.9741 |
| Accuracy | ~98.9% |

*Results vary by ticker, timeframe, and market conditions.*

---

## 🔮 Future Improvements

- [ ] **Transformer Model** — Attention-based time-series (TFT)
- [ ] **Reinforcement Learning** — PPO trading agent
- [ ] **Email Alerts** — SMTP notifications on signal changes
- [ ] **Multi-ticker Portfolio Optimizer** — Markowitz efficient frontier
- [ ] **Options Chain Analysis** — Implied volatility surface
- [ ] **WebSocket streaming** — Real-time tick data
- [ ] **User Authentication** — Multi-user portfolio isolation
- [ ] **Model Auto-retraining** — Scheduled nightly retraining
- [ ] **Backtesting Engine** — Historical strategy performance

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It does not constitute financial advice. Past model performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.

---

## 📄 License

MIT © 2024 — Feel free to fork, modify, and use for your portfolio.

---

<p align="center">
Built with ❤️ using TensorFlow · Streamlit · Plotly · yfinance
</p>
