import pandas as pd

def read_company_data(csv_file='data/companies.csv'):
    """Read and analyze company data from CSV"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Get total count of tickers
        total_tickers = len(df)
        
        # Get count by sector
        sector_counts = df['sector'].value_counts()
        
        # Get count by exchange
        exchange_counts = df['exchange'].value_counts()
        
        # Print analysis
        print(f"\nTotal number of tickers: {total_tickers}")
        
        print("\nTickers by sector:")
        for sector, count in sector_counts.items():
            if pd.notna(sector):  # Check if sector is not NaN
                print(f"{sector}: {count}")
        
        print("\nTickers by exchange:")
        for exchange, count in exchange_counts.items():
            if pd.notna(exchange):  # Check if exchange is not NaN
                print(f"{exchange}: {count}")
        
        # print("\nAll tickers:")
        # for ticker, name in zip(df['ticker'], df['company name']):
        #     print(f"{ticker}: {name}")
            
        return df
        
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None

def main():
    print("Reading company data from CSV...")
    company_data = read_company_data()
    
    if company_data is not None:
        print("\nData analysis complete!")

if __name__ == "__main__":
    main()
