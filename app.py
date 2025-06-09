import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import requests
from datetime import datetime

# ---- PAGE CONFIG ----
st.set_page_config(page_title="AI Financial Advisor", layout="wide")
st.markdown("""
    <style>
        .metric-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 1rem;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.title("📉 AI Financial Advisor")
st.write("Forecast stocks & mutual funds, get investment signals, and calculate your returns – with AI.")
st.markdown("---")

# ---- SIDEBAR INPUT ----
st.sidebar.title("📍 Select Investment")
option_type = st.sidebar.radio("Investment Type", ["Stock", "Mutual Fund"])

# Load full mutual fund list from MFAPI
@st.cache_data(ttl=86400)
def fetch_all_mutual_funds():
    response = requests.get("https://api.mfapi.in/mf")
    return response.json()

def search_mutual_fund_by_name(name_query):
    all_funds = fetch_all_mutual_funds()
    results = [f for f in all_funds if name_query.lower() in f['schemeName'].lower()]
    return results

# Stock or Mutual Fund input
if option_type == "Stock":
    ticker_input = st.sidebar.text_input("🔎 Enter Stock Ticker (e.g. RELIANCE.NS, INFY.BO)", "")
    is_mutual_fund = False
    if not ticker_input:
        st.warning("⚠️ Please enter a valid stock ticker like `RELIANCE.NS`")
else:
    search_name = st.sidebar.text_input("🔎 Search Mutual Fund by Name")
    matched_funds = search_mutual_fund_by_name(search_name) if search_name else []
    selected_scheme = st.sidebar.selectbox("Select Mutual Fund", matched_funds, format_func=lambda x: x['schemeName']) if matched_funds else None
    ticker_input = selected_scheme['schemeCode'] if selected_scheme else None
    is_mutual_fund = True

st.sidebar.markdown("""
---
**Ticker Instructions:**
- For Indian stocks: Use NSE/BSE tickers like `RELIANCE.NS`, `INFY.BO`
- For Mutual Funds: Search by name
---
""")

invest_amt = st.sidebar.number_input("💰 Enter Investment Amount (₹)", value=2000, step=500)
tenure_months = st.sidebar.slider("⏳ Investment Tenure (months)", 1, 24, 6)
expected_return = st.sidebar.slider("📈 Expected Annual Return (%)", 5, 20, 12)

# SIP CALCULATOR
st.sidebar.subheader("🗕️ SIP Calculator")
sip_amount = st.sidebar.number_input("Monthly SIP Amount (₹)", value=1000, step=100)
sip_months = st.sidebar.slider("Duration (Months)", 6, 60, 12)

monthly_rate = (expected_return / 100) / 12
if monthly_rate > 0:
    sip_value = sip_amount * (((1 + monthly_rate) ** sip_months - 1) / monthly_rate) * (1 + monthly_rate)
else:
    sip_value = sip_amount * sip_months
st.sidebar.success(f"Projected SIP Value: ₹ {sip_value:.2f}")

# ---- LOAD DATA ----
@st.cache_data(ttl=60)
def load_stock_data(ticker):
    df = yf.Ticker(ticker).history(period="2y")
    df.reset_index(inplace=True)
    df["Date"] = df["Date"].dt.tz_localize(None)
    return df

@st.cache_data(ttl=60)
def load_mutual_fund_data(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data["data"])
    df = df.rename(columns={"nav": "Close", "date": "Date"})
    df["Close"] = df["Close"].astype(float)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date")
    return df

# ---- SIGNAL LOGIC ----
def get_signal(forecast_df):
    recent = forecast_df.tail(7)
    slope = recent["yhat"].iloc[-1] - recent["yhat"].iloc[0]
    percent_change = (slope / recent["yhat"].iloc[0]) * 100
    threshold = 0.5  # half a percent
    if percent_change > threshold:
        return f"📈 Strong BUY Signal (+{percent_change:.2f}%)"
    elif percent_change < -threshold:
        return f"🔷 Strong SELL Signal ({percent_change:.2f}%)"
    else:
        return f"⚖️ Hold / No Clear Signal ({percent_change:.2f}%)"

# ---- MAIN LOGIC ----
if ticker_input:
    try:
        if is_mutual_fund:
            data = load_mutual_fund_data(ticker_input)
        else:
            data = load_stock_data(ticker_input)

        st.subheader(f"📉 Historical Price for `{ticker_input}`")
        st.line_chart(data.set_index("Date")["Close"])

        # Forecasting
        df = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Forecast chart
        st.subheader("🔮 Forecast for the Next 30 Days")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='blue')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
        ax.set_title("Forecasted Price (Next 30 Days)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (₹)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        latest_price = df["y"].iloc[-1]
        predicted_price = forecast["yhat"].iloc[-1]
        signal = get_signal(forecast)

        col1, col2, col3 = st.columns(3)
        col1.metric("💵 Current Price", f"₹ {latest_price:.2f}")
        col2.metric("🧠 Predicted (30 Days)", f"₹ {predicted_price:.2f}")
        col3.metric("📈 Signal", signal)

        # MAE chart
        try:
            df_test = df[-60:-30]
            model_test = Prophet()
            model_test.fit(df[:-30])
            future_test = model_test.make_future_dataframe(periods=30)
            forecast_test = model_test.predict(future_test)

            y_true = df_test["y"].values
            y_pred = forecast_test.tail(30)["yhat"].values
            mae = mean_absolute_error(y_true, y_pred)

            st.subheader("📊 Model Accuracy - Actual vs Predicted")
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(df_test["ds"].values, y_true, label="Actual Price", marker="o", color="green")
            ax2.plot(df_test["ds"].values, y_pred, label="Predicted Price", marker="x", linestyle="--", color="red")
            ax2.set_title("Model Accuracy (Past 30 Days)")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Price (₹)")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)
            st.info(f"Mean Absolute Error (MAE): ₹ {mae:.2f}")

        except Exception:
            st.warning("⚠️ Not enough historical data for accuracy chart.")

    except Exception as e:
        st.error(f"❌ Error fetching or processing data: {e}")
