import nltk
#remove the line before after running it for the first time
nltk.download('vader_lexicon')
import numpy as np
import pandas as pd
from newsapi import NewsApiClient
from datetime import date, timedelta, datetime

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

pd.set_option('display.max_colwidth', 1000)  # Set max column width for easier readability

NEWS_API_KEY = 'af4db6bbbcf84b8c8974b38dea460ffb'

def get_articles_sentiments(keywrd, startd, sources_list=None, show_all_articles=False):
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    
    if type(startd) == str:
        my_date = datetime.strptime(startd, '%d-%b-%Y')
    else:
        my_date = startd

    # Fetching articles based on keyword, date, and sources
    if sources_list:
        articles = newsapi.get_everything(q=keywrd,
                                          from_param=my_date.isoformat(),
                                          to=(my_date + timedelta(days=1)).isoformat(),
                                          language="en",
                                          sources=",".join(sources_list),
                                          sort_by="relevancy",
                                          page_size=100)
    else:
        articles = newsapi.get_everything(q=keywrd,
                                          from_param=my_date.isoformat(),
                                          to=(my_date + timedelta(days=1)).isoformat(),
                                          language="en",
                                          sort_by="relevancy",
                                          page_size=100)

    date_sentiments = {}
    date_sentiments_list = []
    seen = set()
    
    # Analyzing the sentiment of each article
    for article in articles['articles']:
        if str(article['title']) in seen:
            continue
        else:
            seen.add(str(article['title']))
            article_content = str(article['title']) + '. ' + str(article['description'])
            sentiment = sia.polarity_scores(article_content)['compound']
            date_sentiments.setdefault(my_date, []).append(sentiment)
            date_sentiments_list.append((sentiment, article['url'], article['title'], article['description']))

    # Sorting the articles by sentiment
    date_sentiments_l = sorted(date_sentiments_list, key=lambda tup: tup[0], reverse=True)
    sent_list = list(date_sentiments.values())[0]
    
    # Creating a DataFrame to display the results neatly
    df = pd.DataFrame(date_sentiments_list, columns=['Sentiment', 'URL', 'Title', 'Description'])
    
    # Displaying results in the console
    print(f"Sentiment Analysis for {keywrd} on {my_date.strftime('%d-%b-%Y')}")
    print("-" * 80)
    print(f"Total articles analyzed: {len(df)}")
    print(f"Top 5 Positive Articles:\n")
    print(df.head(5))
    
    # Optionally, print all articles if show_all_articles is set to True
    if show_all_articles:
        print("\nAll Articles Sentiment Analysis:")
        print(df)
    
    return df  # Return the DataFrame for further analysis or saving

# Example usage:
if __name__ == "__main__":
    # Example of using the function
    result_df = get_articles_sentiments('technology', '01-Nov-2024', show_all_articles=True)
