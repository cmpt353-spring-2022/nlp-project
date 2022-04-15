import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline,  make_pipeline
import seaborn as sns
import matplotlib.ticker as mticker
import pymannkendall as mk
import analyze

combined = pd.read_csv("data/filtered.csv.gz")
X_train, X_test, y_train, y_test = train_test_split(combined.title, combined.clickbait)

print()

# Naive baise Multinomial
model_NB_M = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)
model_NB_M.fit(X_train, y_train)

# # Naive baise Bernoulli
model_NB_B = make_pipeline(
     TfidfVectorizer(),
     BernoulliNB()
 )

model_NB_B.fit(X_train, y_train)

# Naive baise Complement
model_NB_C = make_pipeline(
    TfidfVectorizer(),
    ComplementNB()
)

model_NB_C.fit(X_train, y_train)

# Perceptron
model_percp = make_pipeline(
    TfidfVectorizer(),
    Perceptron(max_iter=50)
)

model_percp.fit(X_train, y_train)


scores = []

scores.append(model_NB_M.score(X_test, y_test))
scores.append(model_NB_B.score(X_test, y_test))
scores.append(model_NB_C.score(X_test, y_test))
scores.append(model_percp.score(X_test, y_test))

names = ['Multinomial', 'Bernoulli', 'Complement', 'Perceptron']
scores_dt = pd.DataFrame(scores, columns=["Scores"])
scores_dt["names"] = names
scores_dt['vectorizer'] = "Td-idf"



### USING HASHING VECTORIZER FOR FEATURE EXTRACTION ###

#Naive baise Multinomial
model_NB_M = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)
model_NB_M.fit(X_train, y_train)

# Naive baise Bernoulli
model_NB_B = make_pipeline(
    CountVectorizer(),
    BernoulliNB()
)
model_NB_B.fit(X_train, y_train)

# Naive baise Complement
model_NB_C = make_pipeline(
    CountVectorizer(),
    ComplementNB()
)
model_NB_C.fit(X_train, y_train)

# Perceptron
model_percp = make_pipeline(
    CountVectorizer(),
    Perceptron(max_iter=50)
)
model_percp.fit(X_train, y_train)

scores = []

scores.append(model_NB_M.score(X_test, y_test))
scores.append(model_NB_B.score(X_test, y_test))
scores.append(model_NB_C.score(X_test, y_test))
scores.append(model_percp.score(X_test, y_test))

model_NB_M.score(X_test, y_test)
model_NB_B.score(X_test, y_test)
model_NB_C.score(X_test, y_test)
model_percp.score(X_test, y_test)

names = ['Multinomial', 'Bernoulli', 'Complement', 'Perceptron']
scores_dt2 = pd.DataFrame(scores, columns=["Scores"])
scores_dt2["names"] = names
scores_dt2['vectorizer'] = "Count"
scores_dt = pd.concat([scores_dt, scores_dt2])

plt.figure()
scores_dt.pivot("names", "vectorizer", "Scores").plot(kind="bar")
plt.ylabel("Model Accuracy Score")
plt.xlabel("Model Names")
plt.title("Different ML Models Accuracy Score with Two Different Vectorizers")
plt.legend(bbox_to_anchor=(1.05, 1.05))
plt.savefig("scikitModels.png", bbox_inches='tight', dpi=72)
### ANALYSIS ###



#groups it by month of the year to get enough data points for statistical testing

# =============================================================================
sub_data_files = ["data/UpliftingNews_lg.csv.gz", 
                "data/nottheonion_lg.csv.gz",\
                 "data/worldnews_lg.csv.gz", 
                 "data/politics_lg.csv.gz", 
                 "data/news_lg.csv.gz",
                 "data/canadapolitics_lg.csv.gz"]
names = ["Uplifting News", 
        "Not The Onion", 
        "World News", 
        "Politics", 
        "News",
        "Canada Politics"]

def predict(subreddit: str, model: Pipeline):
    df = pd.read_csv(subreddit)
    df = df.dropna(subset=['title'])
    if ('created' in df.columns):
        df['created'] = pd.to_datetime(df["created"], unit="s")
    else:
        df['created'] = df['created_utc']
    df['year'] = pd.DatetimeIndex(df['created']).year
    df['Y_M'] = pd.DatetimeIndex(df['created']).to_period("M")
    df['prediction'] = model.predict(df.title)
    return df

plt.figure()
for file, names in zip(sub_data_files, names):
    df = predict(file, model_NB_C)
    output, test_res = analyze.get_cb_ratio(df)
    print(test_res)
    plt.plot(output.year, output.cb_ratio, label=names)


sns.set_theme()

plt.legend( bbox_to_anchor=(1.05, 1),loc="upper left")
plt.xlabel("Year")
plt.ylabel("% of clickbait titles")
plt.title("Percentage of clickbait titles in selected news subreddits over the years")
plt.savefig("baitPercentage.png", dpi=72, bbox_inches='tight')



plt.figure(dpi=1200)

df_all = pd.DataFrame()
sub_data_files = ["data/UpliftingNews_lg.csv.gz", 
                "data/nottheonion_lg.csv.gz",\
                 "data/worldnews_lg.csv.gz", 
                 "data/politics_lg.csv.gz", 
                 "data/news_lg.csv.gz",
                 "data/canadapolitics_lg.csv.gz"]
names = ["Uplifting News", 
        "Not The Onion", 
        "World News", 
        "Politics", 
        "News",
        "Canada Politics"]

for file, name in zip(sub_data_files, names):
    data = pd.read_csv(file)
    data = data.dropna(subset=['title'])
    data["prediction"] = model_NB_B.predict(data.title)
    df_res = analyze.ratio_by_score(data, 4)
    df_res['subreddit'] = name
    df_all = pd.concat([df_all, df_res])

df_all = df_all.reset_index()
print(df_all)
df_all.pivot("subreddit", "index", "prediction").plot(kind="bar")
plt.ylabel("Clickbait Percentage (%)")
plt.xlabel("Subreddit")
plt.title("Clickbait Rate per Upvotes Quartile for Selected Subreddits")
plt.legend(["0-25%", "25%-50%", "50%-75%", "75%-100%"], title="Upvotes Quartile",  bbox_to_anchor=(1.05, 1.05))

plt.savefig("scoreVsClickbait.png", bbox_inches='tight', dpi=72)





















