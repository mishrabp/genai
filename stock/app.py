import yfinance as yf
import pandas as pd
import numpy as np
from twilio.rest import Client

# Twilio credentials (replace with your actual Twilio details)
TWILIO_PHONE_NUMBER = "+18557586615"
MY_PHONE_NUMBER = "+12145324096"

# Function to calculate RSI14 manually
def calculate_rsi(data, window=14):
    delta = data.diff(1)  # Price changes
    gain = np.where(delta > 0, delta, 0)  # Positive gains
    loss = np.where(delta < 0, -delta, 0)  # Negative losses

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1]  # Latest RSI value

# Fetch real-time market data
def get_market_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")  # Get 3 months of data for better accuracy

    # Fetch MSA20 and MSA50, handling NaN values
    msa20 = hist["Close"].rolling(window=20).mean().iloc[-1]
    msa50 = hist["Close"].rolling(window=50).mean().iloc[-1]

    if pd.isna(msa20):
        msa20 = hist["Close"].mean()
    if pd.isna(msa50):
        msa50 = hist["Close"].mean()

    # Fetch VIX data from Yahoo Finance
    vix_stock = yf.Ticker("^VIX")
    vix_hist = vix_stock.history(period="1d")
    vix = vix_hist["Close"].iloc[-1] if not vix_hist.empty else 20  # Default to 20 if unavailable

    # Calculate RSI14
    rsi14 = calculate_rsi(hist["Close"], window=14)

    return rsi14, msa20, msa50, vix

# Determine trading action
def get_trading_advice(rsi14, vix):
    if rsi14 > 70:
        return "Overbought! Advise: SELL your holdings."
    elif rsi14 < 30 and vix < 18:
        return "Oversold! Advise: BUY BIG ($1000)."
    elif 30 <= rsi14 <= 60 and vix < 18:
        return "Normal market! Advise: SIP ($100)."
    else:
        return "No clear signal. Stay put."

# Send SMS via Twilio
def send_sms(message):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    sms = client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=MY_PHONE_NUMBER
    )
    print("SMS sent:", sms.sid)

# Main execution
if __name__ == "__main__":
    ticker = input("Enter stock ticker: ").strip().upper()
    rsi14, msa20, msa50, vix = get_market_data(ticker)
    advice = get_trading_advice(rsi14, vix)

    print("\nMarket Indicators:")
    print(f"RSI14: {rsi14:.2f}, MSA20: {msa20:.2f}, MSA50: {msa50:.2f}, VIX: {vix:.2f}")
    print(f"{ticker} is : {advice}")

    # Send SMS with advice
    send_sms(f"{ticker} is : {advice}")
