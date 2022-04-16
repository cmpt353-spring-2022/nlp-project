import pandas as pd
import sys
import typer

import spacy
from spacy.tokens import DocBin


def main(training_ratio: float, path_to_training: str):
    assert type(training_ratio) == float    
    print(f"Using training ratio of {training_ratio}")
    data = pd.read_csv(path_to_training)
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
