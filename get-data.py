from psaw import PushshiftAPI
import pandas as pd
import sys

"""
    DOMAINS:
        1. apnews.com
        2. npr.org
        3. pbs.org
        4. reuters.com

    SUBREDDITS:
        1. savedyouaclick
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
            limit=100000,
        )
    elif (sys.argv[1] == 'subreddit'):
        gen = api.search_submissions(
            subreddit=sys.argv[2],
            filter=["id", "full_link", "created_utc", "url", "title", "subreddit", "author", "upvote_ratio", "score"],
            limit=100000,
        )

    output_file = sys.argv[3] + ".csv.gz"

    df = pd.DataFrame([thing.d_ for thing in gen])
    df.to_csv(output_file, index=False, compression="gzip")

if __name__ == "__main__":
    main()
