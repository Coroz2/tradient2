NEWS_API_KEY = "af4db6bbbcf84b8c8974b38dea460ffb"

import nltk
#remove the line before after running it for the first time
nltk.download('vader_lexicon')
import numpy as np
import pandas as pd
from newsapi import NewsApiClient
from datetime import date, timedelta, datetime
import os

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

pd.set_option('display.max_colwidth', 1000)  # Set max column width for easier readability


def get_articles_sentiments(keywrd, startd, sources_list=None, show_all_articles=False):
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    print(os.getenv('NEWS_API_KEY'))
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
    sentiment_scores = []

    # Analyzing the sentiment of each article
    for article in articles['articles']:
        if str(article['title']) in seen:
            continue
        else:
            seen.add(str(article['title']))
            article_content = str(article['title']) + '. ' + str(article['description'])
            sentiment = sia.polarity_scores(article_content)
            sentiment_scores.append(sentiment['compound'])
            date_sentiments.setdefault(my_date, []).append(sentiment)
            date_sentiments_list.append((sentiment, article['url'], article['title'], article['description']))

    # Calculate the average sentiment score
    avg_sentiment_score = np.mean(sentiment_scores)

    # Create a DataFrame to display the results neatly
    df = pd.DataFrame(date_sentiments_list, columns=['Sentiment', 'URL', 'Title', 'Description'])
    
    print(avg_sentiment_score)
    return df, avg_sentiment_score

# Example usage:
if __name__ == "__main__":
    # Example of using the function
    result_df = get_articles_sentiments('Tesla Stock', '01-Nov-2024', show_all_articles=True)