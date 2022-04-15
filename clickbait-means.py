from scipy import stats
import pandas as pd
import sys
import os
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib import pyplot as plt


def main():
    prediction_path = sys.argv[1]

    files = os.listdir(prediction_path)

    data = [
        pd.read_csv(f"{prediction_path}/{file}").sample(frac=1).reset_index(drop=True)
        for file in files
    ]

    min_len = min(len(df) for df in data)

    predictions = [df["prediction"][:min_len] for df in data]
    anova = stats.f_oneway(*predictions)

    if anova.pvalue >= 0.05:
        print("Failed to reject means being different")
        return

    labelled = pd.DataFrame(
        {file: prediction for file, prediction in zip(files, predictions)}
    )

    melted = pd.melt(labelled)

    print("Difference in clickbait means!")

    posthoc = pairwise_tukeyhsd(melted["value"], melted["variable"], alpha=0.05)

    print(posthoc)

    fig, ax = plt.subplots()

    posthoc.plot_simultaneous(ax=ax)
    
    plt.show()


if __name__ == "__main__":
    main()
