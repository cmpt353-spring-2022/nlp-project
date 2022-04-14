import pandas as pd
import pymannkendall as mk

def get_p_value(subreddit: pd.DataFrame):
    cb_pred = subreddit[subreddit.prediction == 1]
    cb_grouped = cb_pred.groupby("Y_M").count()
    count = subreddit.groupby("Y_M").count()
    count['cb_ratio'] = cb_grouped["prediction"] / count["prediction"]
    count = count[count.created_utc > 100]
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
    count = count[count.created_utc > 500]

    p_value = get_p_value(subreddit)
    return count, p_value



