import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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
        balance_sheet = stock.balance_sheet
        income_statement = stock.financials
        cash_flow = stock.cashflow
        return balance_sheet, income_statement, cash_flow
    except Exception as e:
        raise Exception(f"Failed to fetch data from Yahoo Finance: {e}")

# Fetch stock news
def fetch_stock_news(ticker):
    sn = StockNews(ticker, save_news=False)
    try:
        news_df = sn.read_rss()
        if news_df.empty:
            return []
        filtered_news = news_df[
            (news_df['title'].str.contains(ticker, case=False, na=False)) |
            (news_df['summary'].str.contains(ticker, case=False, na=False))
        ]
        return filtered_news if not filtered_news.empty else news_df
    except Exception as e:
        return []

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Adj Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Sentiment indicator images
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
    if not ticker.strip():
        st.error("Please enter a valid stock ticker symbol.")
        return

    start_date = st.sidebar.date_input("Start Date", value=datetime(2024, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime(2024, 11, 23))
    chart_type = st.sidebar.selectbox("Select Chart Type", ["Line Chart", "Bar Chart", "Candlestick Chart"])

    # Fetch stock data
    extended_start_date = start_date - timedelta(days=14)
    try:
        data = yf.download(ticker, start=extended_start_date, end=end_date)

        if data.empty:
            st.warning("No data available for the selected ticker and date range.")
            return

        # Reset index so 'Date' is an actual column
        data.reset_index(inplace=True)
        data['Date'] = data['Date'].astype(str)  # Convert for Plotly compatibility

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    # Chart rendering
    if chart_type == "Line Chart":
        fig = px.line(data, x="Date", y="Adj Close", title=f"{ticker} - Line Chart")
        st.plotly_chart(fig)

    elif chart_type == "Bar Chart":
        fig = px.bar(data, x="Date", y="Adj Close", title=f"{ticker} - Bar Chart")
        st.plotly_chart(fig)

    elif chart_type == "Candlestick Chart":
        fig = go.Figure(
            data=[go.Candlestick(
                x=data["Date"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Adj Close"]
            )]
        )
        fig.update_layout(title=f"{ticker} - Candlestick Chart")
        st.plotly_chart(fig)

    # Tabs
    tabs = st.tabs(["Pricing Data", "Fundamental Data", "News", "Sentiment Indicator", "Mpulse Chatbot"])

    with tabs[0]:  # Pricing Data
        st.write(f"Pricing Data for {ticker}")
        st.write(data.describe())

    with tabs[1]:  # Fundamental Data
        st.write(f"Fundamental Data for {ticker}")
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
        st.write(f"News for {ticker}")
        try:
            news_df = fetch_stock_news(ticker)
            if news_df.empty:
                st.warning(f"No news articles available for {ticker}.")
            else:
                for i in range(min(10, len(news_df))):
                    st.subheader(f"{i + 1}. {news_df['title'][i]}")
                    st.write(news_df['published'][i])
                    st.write(news_df['summary'][i])
                    if 'link' in news_df.columns and news_df['link'][i]:
                        st.markdown(f"[Read more]({news_df['link'][i]})", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to fetch news for {ticker}. Error: {e}")

    with tabs[3]:  # Sentiment Indicator
        st.write(f"Sentiment Indicator for {ticker}")
        st.write(data['Adj Close'].rolling(14).mean())  # Kept your logic

    with tabs[4]:  # Mpulse Chatbot
        st.write("Ask about stock performance!")
        st.text_input("You:", key="user_input")

# Run the app
if __name__ == "__main__":
    marketpulse()
