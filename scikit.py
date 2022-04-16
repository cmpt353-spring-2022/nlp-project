from turtle import tilt, title
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline,  make_pipeline
import seaborn as sns
import matplotlib.ticker as mticker
import pymannkendall as mk
import analyze
import statsmodels.api as sm
from scipy import stats

combined = pd.read_csv("data/training/training_cleaned.csv.gz")
X_train, X_test, y_train, y_test = train_test_split(combined.title, combined.clickbait)

print()

# Naive baise Multinomial
model_NB_M = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)
model_NB_M.fit(X_train, y_train)

# # Naive baise Bernoulli
model_NB_SGD = make_pipeline(
     TfidfVectorizer(),
     SGDClassifier()
 )

model_NB_SGD.fit(X_train, y_train)

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
scores.append(model_NB_SGD.score(X_test, y_test))
scores.append(model_NB_C.score(X_test, y_test))
scores.append(model_percp.score(X_test, y_test))

names = ['Multinomial', 'SGD', 'Complement', 'Perceptron']
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
model_NB_SGD = make_pipeline(
    CountVectorizer(),
    SGDClassifier()
)
model_NB_SGD.fit(X_train, y_train)

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
scores.append(model_NB_SGD.score(X_test, y_test))
scores.append(model_NB_C.score(X_test, y_test))
scores.append(model_percp.score(X_test, y_test))

names = ['Multinomial', 'SGD', 'Complement', 'Perceptron']
scores_dt2 = pd.DataFrame(scores, columns=["Scores"])
scores_dt2["names"] = names
scores_dt2['vectorizer'] = "Count"
scores_dt = pd.concat([scores_dt, scores_dt2])

plt.figure()
scores_dt.pivot("names", "vectorizer", "Scores").plot(kind="bar")
plt.ylabel("Model Accuracy Score")
plt.xlabel("Model Names")
plt.title("Different ML Models Accuracy Score with Two Different Vectorizers")
plt.legend(bbox_to_anchor=(1.05, 1.05),title="Vectorizer")
plt.savefig("scikitModels.png", bbox_inches='tight', dpi=72)
### ANALYSIS ###



#groups it by month of the year to get enough data points for statistical testing

# =============================================================================
sub_data_files = ["data/classification/upliftingnews.csv.gz", 
                "data/classification/nottheonion.csv.gz",\
                 "data/classification/worldnews.csv.gz", 
                 "data/classification/politics.csv.gz", 
                 "data/classification/news.csv.gz",
                 "data/classification/canadapolitics.csv.gz"]
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
plt.title("Scikit ComplementNB Classifier")
plt.savefig("baitPercentage.png", dpi=72, bbox_inches='tight')



plt.figure(dpi=1200)

df_all = pd.DataFrame()



#For plotting
for file, name in zip(sub_data_files, names):
    data = pd.read_csv(file)
    data = data.dropna(subset=['title'])
    data["prediction"] = model_NB_C.predict(data.title)
    df_res = analyze.ratio_by_score(data, 4)
    regress = analyze.ratio_by_score(data,30)
    reg = stats.linregress(regress["prediction"], regress['score'])
    print(file, ": p-value ",  reg.pvalue, ", Slope: ", reg.slope)
    df_res['subreddit'] = name
    df_all = pd.concat([df_all, df_res])

df_all = df_all.reset_index()
print(df_all)
df_all.pivot("subreddit", "index", "prediction").plot(kind="bar")
plt.ylabel("Clickbait Percentage (%)")
plt.xlabel("Subreddit")
plt.title("Clickbait Rate per Score Quartile for Selected Subreddits")
plt.legend(["0-25%", "25%-50%", "50%-75%", "75%-100%"], title="Score Quartile",  bbox_to_anchor=(1.05, 1.05))

plt.savefig("scoreVsClickbait.png", bbox_inches='tight', dpi=72)



















