import yfinance as yf

ticker_symbol = "AAPL"

ticker = yf.Ticker(ticker_symbol)

historical_data = ticker.history(period="1y")
print("Historical Data:")
print(historical_data)

financials = ticker.financials
print("\nFinancials:")
print(financials)

actions = ticker.actions
print("\nStock Actions:")
print(actions)