import pandas as pd
import sys
import typer

import spacy
from spacy.tokens import DocBin


def main(training_ratio: float, use_filtered: bool = True):
    assert type(training_ratio) == float    
    print(f"Using training ratio of {training_ratio}")

    if (not use_filtered):
        clickbait = pd.read_csv("../clickbait.csv.gz")
        clickbait["created"] = pd.to_datetime(clickbait["created"], unit="s")
        clickbait["title"] = clickbait["title"].str.split("|").str.get(0)
        clickbait = clickbait[clickbait["score"] > 5]
        clickbait["clickbait"] = 1

        clickbait = clickbait[clickbait["title"].notna()]

        apnews = pd.read_csv("../apnews.csv.gz")
        apnews = apnews[apnews["title"].notna()]
        apnews["created"] = pd.to_datetime(apnews["created"], unit="s")
        apnews["clickbait"] = 0
        apnews = apnews.sample(frac=1).reset_index(drop=True)

        assert len(clickbait) < len(apnews)

        training_count = int(training_ratio * len(clickbait))

        train_data = pd.concat(
            [apnews[:training_count], clickbait[:training_count]], ignore_index=True
        )
        train_data = train_data.sample(frac=1).reset_index(drop=True)

        valid_data = pd.concat(
            [apnews[training_count : len(clickbait)], clickbait[training_count:]],
            ignore_index=True,
        )
        valid_data = valid_data.sample(frac=1).reset_index(drop=True)
    else:
        data = pd.read_csv('../filtered.csv.gz')

        training_count = int(training_ratio * len(data))
        
        train_data = data[:training_count]
        valid_data = data[training_count:]

    def to_corpus(data: pd.DataFrame, filename: str):
        nlp = spacy.blank("en")
        db = DocBin()
        for i, row in data.iterrows():
            doc = nlp.make_doc(row["title"])
            if row["clickbait"]:
                doc.cats = {"clickbait": 1.0, "other": 0.0}
            else:
                doc.cats = {"clickbait": 0.0, "other": 1.0}

            db.add(doc)

        db.to_disk(f"corpus/{filename}.spacy")

    to_corpus(train_data, "train")
    to_corpus(valid_data, "dev")


if __name__ == "__main__":
    typer.run(main)
