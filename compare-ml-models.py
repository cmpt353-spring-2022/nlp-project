from pathlib import Path
from scipy import stats
import pandas as pd
import sys
import os
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib import pyplot as plt


prediction_path = "./data/predictions"


def to_abs_path(name, path):
    return f"{prediction_path}/{name}/{path}"


# because NLTK data is filtered
def filter_data(df: pd.DataFrame):
    if "score" in df.columns:  # because NLTK does not have a score column
        return df[df["score"] > 50]
    return df


def concat_files_to_df(files):
    return pd.concat((filter_data(pd.read_csv(f)) for f in files)).reset_index(
        drop=True
    )


def main():
    spacy = concat_files_to_df(
        map(lambda f: to_abs_path("spacy", f), os.listdir(f"{prediction_path}/spacy"))
    )
    multinomial = concat_files_to_df(
        map(
            lambda f: to_abs_path("scikit/multinomial", f),
            os.listdir(f"{prediction_path}/scikit/multinomial"),
        )
    )
    sgd = concat_files_to_df(
        map(
            lambda f: to_abs_path("scikit/sgd", f),
            os.listdir(f"{prediction_path}/scikit/sgd"),
        )
    )
    perceptron = concat_files_to_df(
        map(
            lambda f: to_abs_path("scikit/perceptron", f),
            os.listdir(f"{prediction_path}/scikit/perceptron"),
        )
    )
    complement = concat_files_to_df(
        map(
            lambda f: to_abs_path("scikit/complement", f),
            os.listdir(f"{prediction_path}/scikit/complement"),
        )
    )
    nltk = concat_files_to_df(
        map(
            lambda f: to_abs_path("", f),
            filter(
                lambda f: f.startswith("nltk_predictions"), os.listdir(prediction_path)
            ),
        )
    )
    nltk = nltk.rename(columns={"nb_predictions": "prediction"})

    min_len = min(len(f) for f in [nltk, complement, sgd, multinomial, spacy])

    # print(len(nltk))
    # print(len(complement))
    # print(len(sgd))
    # print(len(multinomial))
    # print(len(spacy))

    assert len(nltk) == len(complement) == len(sgd) == len(multinomial) == len(spacy)

    data = [spacy, multinomial, sgd, perceptron, complement, nltk]
    labels = ["spacy", "multinomial", "sgd", "perceptron", "complement", "nltk"]

    predictions = [df["prediction"][:min_len] for df in data]
    anova = stats.f_oneway(*predictions)

    if anova.pvalue >= 0.05:
        print("Failed to reject means of model predictions being different")
        return

    print(labels)

    labelled = pd.DataFrame(
        {file: prediction for file, prediction in zip(labels, predictions)}
    )

    melted = pd.melt(labelled)

    print("Difference in prediction means!")

    posthoc = pairwise_tukeyhsd(melted["value"], melted["variable"], alpha=0.05)

    print(posthoc)

    fig, ax = plt.subplots()

    posthoc.plot_simultaneous(ax=ax)

    plt.show()


if __name__ == "__main__":
    main()
