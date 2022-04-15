import os
import spacy
import typer
import en_textcat_clickbait
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import analyze
from pathlib import Path

sub_data_files = [
    "../data/UpliftingNews_lg.csv.gz",
    "../data/nottheonion_lg.csv.gz",
    "../data/worldnews_lg.csv.gz",
    "../data/politics_lg.csv.gz",
    "../data/news_lg.csv.gz",
    "../data/canadapolitics_lg.csv.gz",
]


def predict(subreddit: str, model: spacy.language.Language, output_file: str):
    print(f"Predicting on {subreddit} with {model.pipe_names}")
    df = pd.read_csv(subreddit)
    df = df.dropna(subset=["title"])
    if "created" in df.columns:
        df["created"] = pd.to_datetime(df["created"], unit="s")
    else:
        df["created"] = df["created_utc"]
    df["year"] = pd.DatetimeIndex(df["created"]).year
    df["Y_M"] = pd.DatetimeIndex(df["created"]).to_period("M")
    df["prediction"] = [
        doc.cats["clickbait"] for doc in model.pipe(df["title"].values, batch_size=2000)
    ]
    print(f"Done prediction on {subreddit}")
    df["prediction"] = df["prediction"].round().astype(int)
    df.to_csv(
        output_file,
        compression="gzip",
        index=False,
    )
    print(f"Wrote predictions to file")
    return df


def main():
    if spacy.prefer_gpu():
        print("Using GPU for predictions")
    else:
        print("Using CPU for predictions")
    nlp = en_textcat_clickbait.load()
    for file in sub_data_files:
        name = Path(Path(file).stem).stem
        output_file = f"predictions/{name}.csv.gz"
        if not os.path.isfile(output_file):
            df = predict(file, nlp, output_file)
        else:
            df = pd.read_csv(output_file)
        output, test_res = analyze.get_cb_ratio(df)
        print(test_res)
        plt.plot(output.year, output.cb_ratio, label=name)
    sns.set_theme()

    plt.legend(loc="upper left")
    plt.xlabel("Year")
    plt.ylabel("% of clickbait titles")
    plt.title(
        "Percentage of clickbait titles in selected news subreddits over the years (Textcat)"
    )
    plt.show()


if __name__ == "__main__":
    typer.run(main)