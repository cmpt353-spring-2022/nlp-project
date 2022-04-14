from psaw import PushshiftAPI
import pandas as pd

api = PushshiftAPI()

gen = api.search_submissions(
    subreddit="savedyouaclick",
    filter=["url", "author", "title", "subreddit", "upvote_ratio", "full_link", "score"],
    limit=100000,
)

df = pd.DataFrame([thing.d_ for thing in gen])

df.to_csv("data/clickbait.csv.gz", index=False, compression="gzip")
