import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer
import seaborn as sns
import matplotlib.ticker as mticker
import pymannkendall as mk



apnews = pd.read_csv("apnews.csv.gz")
apnews["title_before_bar"] = apnews.title.str.split(r'|').str.get(0)
apnews["title_after_bar"] = apnews.title.str.split(r'|').str.get(1)
apnews["category"] = "not_a_clickbait"


clickbait = pd.read_csv("clickbait.csv.gz")
clickbait["title_before_bar"] = clickbait.title.str.split(r'|').str.get(0)
clickbait["title_after_bar"] = clickbait.title.str.split(r'|').str.get(1)
clickbait["category"] = "clickbait"

combined = pd.concat([apnews, clickbait])
X_train, X_test, y_train, y_test = train_test_split(combined.title_before_bar, combined.category)

print()

# Naive baise Multinomial
# model_NB_M = make_pipeline(
#     TfidfVectorizer(),
#     MultinomialNB()
# )

# model_NB_M.fit(X_train, y_train)

# # Naive baise Bernoulli
# model_NB_B = make_pipeline(
#     TfidfVectorizer(),
#     BernoulliNB()
# )

# model_NB_B.fit(X_train, y_train)

# # Naive baise Complement
# model_NB_C = make_pipeline(
#     TfidfVectorizer(),
#     ComplementNB()
# )

# model_NB_C.fit(X_train, y_train)

# Perceptron
model_percp = make_pipeline(
    TfidfVectorizer(),
    Perceptron(max_iter=50)
)

model_percp.fit(X_train, y_train)

# print("Using Term Frequency–Inverse Document Frequency")

# model_NB_M.score(X_test, y_test)
# print("Multinomial", model_NB_M.score(X_test, y_test))

# model_NB_B.score(X_test, y_test)
# print("Bernoulli", model_NB_B.score(X_test, y_test))

# model_NB_C.score(X_test, y_test)
# print("Complement", model_NB_C.score(X_test, y_test))

# model_percp.score(X_test, y_test)



# answer = model_NB_M.predict(["Lex Fridman at Tesla Giga Texas grand opening"])
# print(answer)



### USING HASHING VECTORIZER FOR FEATURE EXTRACTION ###

#Naive baise Multinomial
# model_NB_M = make_pipeline(
#     CountVectorizer(),
#     Binarizer(),
#     MultinomialNB()
# )
# model_NB_M.fit(X_train, y_train)

# # Naive baise Bernoulli
# model_NB_B = make_pipeline(
#     CountVectorizer(),
#     Binarizer(),
#     BernoulliNB()
# )
# model_NB_B.fit(X_train, y_train)

# # Naive baise Complement
# model_NB_C = make_pipeline(
#     CountVectorizer(),
#     Binarizer(),
#     ComplementNB()
# )
# model_NB_C.fit(X_train, y_train)

# # Perceptron
# model_percp = make_pipeline(
#     CountVectorizer(),
#     Binarizer(),
#     Perceptron(max_iter=50)
# )
# model_percp.fit(X_train, y_train)

# print()
# print("Using One-hot Encoding")

# model_NB_M.score(X_test, y_test)
# print("Multinomial", model_NB_M.score(X_test, y_test))

# model_NB_B.score(X_test, y_test)
# print("Bernoulli", model_NB_B.score(X_test, y_test))

# model_NB_C.score(X_test, y_test)
# print("Complement", model_NB_C.score(X_test, y_test))

# model_percp.score(X_test, y_test)
# print("Perceptron", model_percp.score(X_test, y_test))




### ANALYSIS ###



#groups it by month of the year to get enough data points for statistical testing

sub_data_files = ["UpliftingNews_lg.csv.gz", "nottheonion_lg.csv.gz"]

def get_p_value(subreddit):
    cb_pred = subreddit[subreddit.prediction == "clickbait"]
    cb_grouped = cb_pred.groupby("Y_M").count()
    count = subreddit.groupby("Y_M").count()
    count['cb_ratio'] = cb_grouped["prediction"] / count["prediction"]
    count = count[count.author > 100]
    count = count.reset_index()
    mk_res = mk.original_test(count['cb_ratio'])
    return mk_res

#grouped by year for visualization
def get_cb_ratio(subreddit):
    subreddit['created'] = pd.to_datetime(subreddit["created"], unit="s")
    subreddit['year'] = pd.DatetimeIndex(subreddit['created']).year
    subreddit['Y_M'] = pd.DatetimeIndex(subreddit['created']).to_period("M")
    subreddit['prediction'] = model_percp.predict(subreddit.title)
    
    
    cb_pred = subreddit[subreddit.prediction == "clickbait"]
    cb_grouped = cb_pred.groupby("year").count()

    count = subreddit.groupby("year").count()
    count['cb_ratio'] = cb_grouped["prediction"] / count["prediction"]
    count = count.reset_index()
    count = count[count.author > 500]

    p_value = get_p_value(subreddit)
    return count, p_value





for file in sub_data_files:
    subreddit = pd.read_csv(file)
    subreddit = subreddit.dropna(subset=['title'])
    output, test_res = get_cb_ratio(subreddit)
    print(test_res)
    plt.plot(output.year, output.cb_ratio)





sns.set_theme()

plt.xlabel("Year")
plt.ylabel("% of clickbait titles")
plt.title("Percentage of clickbait titles in selected news subreddits over the years")
plt.show()



































