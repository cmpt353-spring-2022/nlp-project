import pandas as pd


def data_summary(df: pd.DataFrame, name: str):
    print(f"{name} date range: {df['created'].min()} - {df['created'].max()}")
    print(f"{name} top score: {df['score'].max()}")
    print(f"{name} count: {len(df)}")
    print(f"{name} average score: {df['score'].mean()}")

    print()


apnews = pd.read_csv("apnews.csv.gz")
apnews["created"] = pd.to_datetime(apnews["created"], unit="s")
data_summary(apnews, "AP news")

clickbait = pd.read_csv("clickbait.csv.gz")
clickbait["created"] = pd.to_datetime(clickbait["created"], unit="s")
data_summary(clickbait, "Clickbait")
