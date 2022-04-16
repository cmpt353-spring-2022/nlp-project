import pandas as pd

def filter_csv(filename):
    df = pd.read_csv(filename)

    df.drop_duplicates(subset='url', inplace=True)
    df.drop_duplicates(subset='title', inplace=True)
    df.drop_duplicates(subset='id', inplace=True)
    df.drop_duplicates(subset='full_link', inplace=True)

    df = df[['created_utc', 'title', 'score']]
    df = df[df['score'] >= 5]
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
    df = df[df['created_utc'].dt.year >= 2014]
    df = df[df['created_utc'].dt.year <= 2021]

    return df

canadapolitics = filter_csv('data/classification/canadapolitics_lg.csv.gz')
news = filter_csv('data/classification/news_lg.csv.gz')
nottheonion = filter_csv('data/classification/nottheonion_lg.csv.gz')
politics = filter_csv('data/classification/politics_lg.csv.gz')
upliftingnews = filter_csv('data/classification/upliftingnews_lg.csv.gz')
worldnews = filter_csv('data/classification/worldnews_lg.csv.gz')

canadapolitics.to_csv('data/classification/canadapolitics.csv.gz', index=False, compression="gzip")
news.to_csv('data/classification/news.csv.gz', index=False, compression="gzip")
nottheonion.to_csv('data/classification/nottheonion.csv.gz', index=False, compression="gzip")
politics.to_csv('data/classification/politics.csv.gz', index=False, compression="gzip")
upliftingnews.to_csv('data/classification/upliftingnews.csv.gz', index=False, compression="gzip")
worldnews.to_csv('data/classification/worldnews.csv.gz', index=False, compression="gzip")
