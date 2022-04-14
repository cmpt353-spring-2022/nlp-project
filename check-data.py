import pandas as pd


def data_summary(df: pd.DataFrame, name: str):
    print(f"{name} date range: {df['created'].min()} - {df['created'].max()}")
    print(f"{name} top score: {df['score'].max()}")
    print(f"{name} count: {len(df)}")
    print(f"{name} average score: {df['score'].mean()}")
    print(f"{name} sample title: {df.sample(n=1).head(n=1).title.values[0]}")

    print()


def process_data(file: str):
    data = pd.read_csv(f"data/{file}.csv.gz")
    data["created"] = pd.to_datetime(data["created"], unit="s")
    data_summary(data, file)


[process_data(x) for x in ["apnews", "clickbait", "npr", "pbs", "reuters"]]
