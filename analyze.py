from re import sub
import numpy as np
import pandas as pd
import pymannkendall as mk

def get_p_value(subreddit: pd.DataFrame):
    cb_pred = subreddit[subreddit.prediction == 1]
    cb_grouped = cb_pred.groupby("Y_M").count()
    count = subreddit.groupby("Y_M").count()
    count['cb_ratio'] = cb_grouped["prediction"] / count["prediction"]
    count = count[count.created_utc > 50]
    count = count.reset_index()
    mk_res = mk.original_test(count['cb_ratio'])
    return mk_res

#grouped by year for visualization
def get_cb_ratio(subreddit: pd.DataFrame):
    cb_pred = subreddit[subreddit.prediction == 1]
    cb_grouped = cb_pred.groupby("year").count()

    count = subreddit.groupby("year").count()
    count['cb_ratio'] = cb_grouped["prediction"] / count["prediction"]
    count = count.reset_index()
    count = count[count.created_utc > 50]

    p_value = get_p_value(subreddit)
    return count, p_value



#clickbait rate by score
def ratio_by_score(subreddit: pd.DataFrame, partitions):
    df = subreddit[subreddit.score > 5]
    length = df.shape[0]
    df = df.sort_values(by="score").reset_index(drop=True)
    df["part_num"] = np.vectorize(partition_num)(df.index, df.shape[0], partitions)
    df = df[df.prediction == 1]
    df = df.groupby(by="part_num").agg(score=('score', 'mean'), prediction=('prediction', 'count')).reset_index(drop=True)
    df["prediction"] = df["prediction"] / (length/partitions)
    return df

def partition_num(index, length, partitions):
    return np.floor(index / (length/partitions))
