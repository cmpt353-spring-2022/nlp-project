from psaw import PushshiftAPI
import pandas as pd
import sys

"""
    DOMAINS USED (TRAINING):
        1. apnews.com
        2. npr.org
        3. pbs.org
        4. reuters.com

    SUBREDDITS USED (TRAINING):
        1. savedyouaclick

    SUBREDDITS USED (CLASSIFICATION):
        1. canadapolitics
        2. news
        3. nottheonion
        4. politics
        5. upliftingnews
        6. worldnews
"""

def main():
    api = PushshiftAPI()

    if (len(sys.argv) != 4 or (sys.argv[1] != 'domain' and sys.argv[1] != 'subreddit')):
        print("USAGE: python get-data.py <'domain' | 'subreddit'> <domain_name | subreddit_name> <output_file>")
        return

    if (sys.argv[1] == 'domain'):
        gen = api.search_submissions(
            domain=sys.argv[2],
            filter=["id", "full_link", "created_utc", "url", "title", "subreddit", "author", "upvote_ratio", "score"],
            limit=1000000,
            score=">5",
        )
    elif (sys.argv[1] == 'subreddit'):
        gen = api.search_submissions(
            subreddit=sys.argv[2],
            filter=["id", "full_link", "created_utc", "url", "title", "subreddit", "author", "upvote_ratio", "score"],
            limit=1000000,
            score=">5",
        )

    output_file = "data/" + sys.argv[3] + ".csv.gz"

    df = pd.DataFrame([thing.d_ for thing in gen])
    df.to_csv(output_file, index=False, compression="gzip")

if __name__ == "__main__":
    main()
