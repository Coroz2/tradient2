NEWS_API_KEY = "af4db6bbbcf84b8c8974b38dea460ffb"

import nltk
# Remove the line below after running it for the first time
nltk.download('vader_lexicon')

import numpy as np
import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import os

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

pd.set_option('display.max_colwidth', 1000)  # Set max column width for easier readability


def get_articles_sentiments(keywrd, show_all_articles=False):
    # Initialize NewsAPI client
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    # Define business-related sources

    # Get today's date in the required format
    my_date = datetime.today()
    to_date = my_date.strftime('%Y-%m-%dT%H:%M:%S')  # Format: YYYY-MM-DDTHH:MM:SS
    from_date = (my_date - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%S')

    # Fetching articles based on keyword, date, and sources
    try:
        articles = newsapi.get_everything(
            q=keywrd,
            from_param=from_date,
            to=to_date,
            language="en",
            sources="business-insider,financial-times,bloomberg,cnbc,forbes",
            sort_by="relevancy",
            page_size=100
        )
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return None, None

    if 'articles' not in articles or not articles['articles']:
        print("No articles found.")
        return pd.DataFrame(), 0

    date_sentiments_list = []
    seen = set()
    sentiment_scores = []

    # Analyzing the sentiment of each article
    for article in articles['articles']:
        if str(article['title']) in seen:
            continue
        seen.add(str(article['title']))
        article_content = f"{article['title']}. {article['description']}"
        sentiment = sia.polarity_scores(article_content)
        sentiment_scores.append(sentiment['compound'])
        date_sentiments_list.append((sentiment, article['url'], article['title'], article['description']))

    # Calculate the average sentiment score
    avg_sentiment_score = np.mean(sentiment_scores) if sentiment_scores else 0

    # Create a DataFrame to display the results neatly
    df = pd.DataFrame(date_sentiments_list, columns=['Sentiment', 'URL', 'Title', 'Description'])

    print("Average Sentiment Score:")
    print(avg_sentiment_score)
    print('df', df.head())

    return df, avg_sentiment_score
