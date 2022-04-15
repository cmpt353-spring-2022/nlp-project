import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Binarizer
import seaborn as sns
import matplotlib.ticker as mticker
import pymannkendall as mk
import analyze

combined = pd.read_csv("data/filtered.csv.gz")
X_train, X_test, y_train, y_test = train_test_split(combined.title, combined.clickbait)

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

# print("Using Term Frequency-Inverse Document Frequency")

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

# =============================================================================
sub_data_files = ["data/UpliftingNews_lg.csv.gz", "data/nottheonion_lg.csv.gz", "data/worldnews_lg.csv.gz", "data/politics_lg.csv.gz", "data/science_lg.csv.gz"]
names = ["Uplifting News", "Not The Onion", "World News", "Politics", "Science"]
# 
# def predict(subreddit: str, model: Pipeline):
#     df = pd.read_csv(subreddit)
#     df = df.dropna(subset=['title'])
#     if ('created' in df.columns):
#         df['created'] = pd.to_datetime(df["created"], unit="s")
#     else:
#         df['created'] = df['created_utc']
#     df['year'] = pd.DatetimeIndex(df['created']).year
#     df['Y_M'] = pd.DatetimeIndex(df['created']).to_period("M")
#     df['prediction'] = model.predict(df.title)
#     # maybe save these predictions, so we don't have to re-run this
#     return df
# 
# 
# for file, names in zip(sub_data_files, names):
#     df = predict(file, model_percp)
#     output, test_res = analyze.get_cb_ratio(df)
#     print(test_res)
#     plt.plot(output.year, output.cb_ratio, label=names)
# 
# 
# sns.set_theme()
# 
# plt.legend( bbox_to_anchor=(1.05, 1),loc="upper left")
# plt.xlabel("Year")
# plt.ylabel("% of clickbait titles")
# plt.title("Percentage of clickbait titles in selected news subreddits over the years")
# plt.show()
# =============================================================================


plt.figure()

df_all = pd.DataFrame()

for file, name in zip(sub_data_files, names):
    df = pd.read_csv(file)
    df = df.dropna(subset=['title'])
    df["prediction"] = model_percp.predict(df.title)
    df_res = analyze.ratio_by_score(df, 4)
    df_res['name'] = name
    df_all = pd.concat([df_all, df_res])
    print(df_all)

df_all = df_all.reset_index()
df_all.pivot("name", "index", "prediction").plot(kind="bar")





















