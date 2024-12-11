import nltk
# Ensure NLTK's VADER Lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

import numpy as np
import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Check if NEWS_API_KEY is set
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
if not NEWS_API_KEY:
    raise EnvironmentError("NEWS_API_KEY environment variable is not set. Please set it and try again.")

# Initialize Sentiment Analyzer
try:
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    raise RuntimeError(f"Error initializing SentimentIntensityAnalyzer: {e}")

# Set Pandas options for readability
pd.set_option('display.max_colwidth', 1000)

def get_articles_sentiments(keywrd, show_all_articles=False):
    """Fetches articles and analyzes their sentiment."""
    print("Keyword for search:", keywrd)

    # Initialize NewsAPI client
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    # Define date range for fetching articles
    my_date = datetime.today()
    to_date = my_date.strftime('%Y-%m-%dT%H:%M:%S')  # Format: YYYY-MM-DDTHH:MM:SS
    from_date = (my_date - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%S')

    try:
        # Fetch articles
        articles = newsapi.get_everything(
            q=keywrd,
            from_param=from_date,
            to=to_date,
            language="en",
            sources="business-insider,financial-times,bloomberg,cnbc,forbes,the-wall-street-journal,reuters,marketwatch," \
                    "the-motley-fool,investopedia,techcrunch,wired,the-economic-times,nikkei,financial-express,fxstreet," \
                    "kitco-news,coindesk,cointelegraph",
            sort_by="relevancy",
            page_size=100
        )
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return None, None

    if not articles or 'articles' not in articles or not articles['articles']:
        print("No articles found.")
        return pd.DataFrame(), 0

    date_sentiments_list = []
    seen = set()
    sentiment_scores = []

    for article in articles['articles']:
        if not article.get('title') or not article.get('description'):
            continue

        title = str(article['title'])
        if title in seen:
            continue

        seen.add(title)
        article_content = f"{title}. {article['description']}"

        try:
            sentiment = sia.polarity_scores(article_content)
        except Exception as e:
            print(f"Error analyzing sentiment for article '{title}': {e}")
            continue

        sentiment_scores.append(sentiment['compound'])
        date_sentiments_list.append((sentiment, article['url'], title, article['description']))


    avg_sentiment_score = np.mean(sentiment_scores)
    df = pd.DataFrame(date_sentiments_list, columns=['Sentiment', 'URL', 'Title', 'Description'])

    print("Average Sentiment Score:", avg_sentiment_score)
    print("Sample DataFrame:")
    print(df.head())

    return df, avg_sentiment_score
