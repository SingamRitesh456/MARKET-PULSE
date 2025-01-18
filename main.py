import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews
import base64
import requests

# Page configuration
st.set_page_config(layout="wide")

# Stock analysis functions
def analyze_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        current_price = data['Close'].iloc[-1]
        moving_average_50 = data['Close'].rolling(window=50).mean().iloc[-1]
        moving_average_200 = data['Close'].rolling(window=200).mean().iloc[-1]
        return {
            "current_price": current_price,
            "moving_average_50": moving_average_50,
            "moving_average_200": moving_average_200
        }
    except Exception as e:
        return {"error": f"Error fetching stock data: {e}"}

# Fetch fundamental data
def fetch_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        # Fetch financial data
        balance_sheet = stock.balance_sheet
        income_statement = stock.financials
        cash_flow = stock.cashflow
        return balance_sheet, income_statement, cash_flow
    except Exception as e:
        raise Exception(f"Failed to fetch data from Yahoo Finance: {e}")

# Fetch stock news
def fetch_stock_news(ticker):
    sn = StockNews(ticker, save_news=False)
    return sn.read_rss()

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Adj Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.squeeze()

# Sentiment indicator images
def load_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def get_rsi_image(rsi):
    if rsi < 30:
        return "GREEN.png"
    elif 30 <= rsi < 40:
        return "LIGHTGREEN.png"
    elif 40 <= rsi < 60:
        return "YELLOW.png"
    elif 60 <= rsi < 70:
        return "ORANGE.png"
    else:
        return "RED.png"

# GROQ AI Configuration
GROQ_API_KEY = "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"
def generate_response(prompt):
    """Generate response using GROQ AI."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers
        )
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "No response from MPULSE AI.")
    except Exception as e:
        return f"Error: {e}"

# MarketPulse dashboard
def marketpulse():
    st.title("MarketPulse - Stock Dashboard")

    # Sidebar
    st.sidebar.title("Options")
    ticker = st.sidebar.text_input("Custom Ticker", value="TSLA")
    start_date = st.sidebar.date_input("Start Date", value=datetime(2024, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime(2024, 11, 23))
    chart_type = st.sidebar.selectbox("Select Chart Type", ["Line Chart", "Bar Chart", "Candlestick Chart"])

    # Fetch stock data
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.warning("No data available for the selected ticker and date range.")
            return
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    # Chart rendering
    if chart_type == "Line Chart":
        fig = px.line(
            data.reset_index(),
            x="Date",
            y=data["Adj Close"].values.flatten(),
            title=f"{ticker} - Line Chart"
        )
    elif chart_type == "Bar Chart":
        fig = px.bar(
            data.reset_index(),
            x="Date",
            y=data["Adj Close"].values.flatten(),
            title=f"{ticker} - Bar Chart"
        )
    elif chart_type == "Candlestick Chart":
        fig = go.Figure(
            data=[go.Candlestick(
                x=data.index,
                open=data["Open"].squeeze(),
                high=data["High"].squeeze(),
                low=data["Low"].squeeze(),
                close=data["Adj Close"].squeeze()
            )]
        )
        fig.update_layout(title=f"{ticker} - Candlestick Chart")
    st.plotly_chart(fig)

    # Tabs
    tabs = st.tabs(["Pricing Data", "Fundamental Data", "News", "Sentiment Indicator", "Mpulse Chatbot"])

    with tabs[0]:  # Pricing Data
        st.write("Pricing Data")
        st.write(data.describe())

    with tabs[1]:  # Fundamental Data
        st.write("Fundamental Data")
        try:
            balance_sheet, income_statement, cash_flow = fetch_fundamental_data(ticker)
            st.subheader("Balance Sheet")
            st.write(balance_sheet)
            st.subheader("Income Statement")
            st.write(income_statement)
            st.subheader("Cash Flow Statement")
            st.write(cash_flow)
        except Exception as e:
            st.error("Failed to fetch fundamental data.")

    with tabs[2]:  # News
        st.write("News")
        try:
            news_df = fetch_stock_news(ticker)
            for i in range(min(10, len(news_df))):
                st.subheader(f"{i + 1}. {news_df['title'][i]}")
                st.write(news_df['published'][i])
                st.write(news_df['summary'][i])
        except Exception as e:
            st.error("Failed to fetch news.")

    with tabs[3]:  # Sentiment Indicator
        st.write("Sentiment Indicator")
        try:
            data['RSI'] = calculate_rsi(data)
            current_rsi = data['RSI'].iloc[-1]
            st.write(f"RSI: {current_rsi:.2f}")

            # Display corresponding image
            image_file = get_rsi_image(current_rsi)
            st.image(image_file, caption=f"Sentiment Indicator: {current_rsi:.2f}")
        except Exception as e:
            st.error("Failed to calculate RSI.")

    with tabs[4]:  # Mpulse Chatbot
        st.title("Mpulse Chatbot")
        st.write("Ask your questions about stock performance, trends, or other topics!")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        user_input = st.text_input("You: ", key="user_input")
        if user_input:
            response = generate_response(user_input)
            st.session_state.chat_history.insert(0, {"user": user_input, "bot": response})

        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Bot:** {chat['bot']}\n---")

# Run the app
if __name__ == "__main__":
    marketpulse()
