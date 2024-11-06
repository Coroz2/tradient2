import pandas as pd

def filter_major_exchanges(input_csv='data/companies.csv', output_csv='data/major_exchanges.csv'):
    """Filter and save only NYSE and Nasdaq Global Select stocks"""
    try:
        # Read the original CSV
        df = pd.read_csv(input_csv)
        
        # Filter for specific exchanges
        major_exchanges = ['New York Stock Exchange', 'Nasdaq Global Select']
        filtered_df = df[df['exchange'].isin(major_exchanges)]
        
        # Save filtered data
        filtered_df.to_csv(output_csv, index=False)
        
        # Print summary
        print("\nFiltered Exchange Statistics:")
        exchange_counts = filtered_df['exchange'].value_counts()
        for exchange, count in exchange_counts.items():
            print(f"{exchange}: {count}")
            
        print(f"\nTotal tickers saved: {len(filtered_df)}")
        print(f"Data saved to: {output_csv}")
        
        return filtered_df
        
    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")
        return None

def main():
    print("Filtering major exchanges...")
    filtered_data = filter_major_exchanges()
    
    if filtered_data is not None:
        print("\nFiltering complete!")

if __name__ == "__main__":
    main()