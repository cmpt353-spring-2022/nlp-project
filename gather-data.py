from http import client
from re import sub
from urllib import request
import pandas as pd
import praw


# For demonstartion purposes only
# In order to re-run the script the user must obtain reddit API's KEY and CLIENT ID


KEY='YOUR-KEY'
CLIENT_ID ='YOUR-CLIENT_ID'


USER_AGENT = '353API/0.0.1'

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret = KEY,
    user_agent=USER_AGENT
)

titles = []
created_at = []
upvotes = []
upvotes_ratio = []


def callApi(category, time_filter='day', limit=None):
    if (category == 'hot' or category == 'new'):
        for submission in getattr(reddit.subreddit('savedyouaclick'), category)(limit=limit):
            titles.append(submission.title)
            created_at.append(submission.created_utc)
            upvotes.append(submission.ups)
            upvotes_ratio.append(submission.upvote_ratio)
    else:
        for submission in getattr(reddit.subreddit('savedyouaclick'), category)(limit=limit, time_filter=time_filter):
            titles.append(submission.title)
            created_at.append(submission.created_utc)
            upvotes.append(submission.ups)
            upvotes_ratio.append(submission.upvote_ratio)


callApi('top', 'day')
callApi('top', 'week')
callApi('top', 'month')
callApi('top', 'year')
callApi('top', 'all')
callApi('controversial', 'day')
callApi('controversial', 'week')
callApi('controversial', 'month')
callApi('controversial', 'year')
callApi('controversial', 'all')
callApi('hot')
callApi('new')

data = pd.DataFrame({"title": titles,
                     'created_at': created_at,
                     'upvotes':upvotes,
                     'upvotes_ratio': upvotes_ratio})

data.title = data.title.str.split(r'|').str.get(0)
data = data.drop_duplicates(subset=['title'])

data.to_csv('clickbait_titles.csv', index=False)