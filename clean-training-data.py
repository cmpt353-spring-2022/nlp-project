import pandas as pd

def clean_csv(filename, clickbait, threshold):
    df = pd.read_csv(filename)

    df.drop_duplicates(subset='url', inplace=True)
    df.drop_duplicates(subset='title', inplace=True)
    df.drop_duplicates(subset='full_link', inplace=True)

    analysis_subreddits = ['canadapolitics', 'news', 'nottheonion', 'politics', 'upliftingnews', 'worldnews']
    for sub in analysis_subreddits:
        df = df[df['subreddit'].str.contains(sub, case=False, regex=False) == False]

    df = df[df['title'].str.contains('savedyouaclick', case=False, regex=False) == False]
    #df['title'] = df['title'].str.strip().str.lower()

    news_orgs = ['forbes', 'cnbc', 'espn', 'bbc', 'huffpo', 'nyt', 'reuters', 'ap news', 'associated press', 'pbs newshour', 'pbs', 'cnn', 'sydney morning herald', 'chicago tribune', 'chicago sun-times', 'la times', 'iol', 'al jazeera', 'washington post', 'toronto star', 'telegraph', 'bbc', 'south china morning post', 'npr', 'guardian', 'the japan times', 'abc', 'fox', 'al arabiya', 'nbc', 'irish times', 'manila bulletin', 'nz herald', 'times of india', 'ap', 'sana', 'usatoday', 'nypost']
    for org in news_orgs:
        df['title'].replace('(?i)\|(.?)' + org + '(.*)$', '', inplace=True, regex=True)
        df['title'].replace('(?i):(.?)' + org + '(.*)$', '', inplace=True, regex=True)
        df['title'].replace('(?i)-(.?)' + org + '(.*)$', '', inplace=True, regex=True)
        df['title'].replace('(?i)^' + org + '(.*):', '', inplace=True, regex=True)
        df['title'].replace('(?i)^breaking(.?)' + org + '(.?):', '', inplace=True, regex=True)

    regs = ['^(?:\[)(.*)(?:\])(.?)-', '^\[ap\]', '^\(ap:\)', '^\[AP news\]', '^news:', '^exclusive:', '^breaking news:', '^the indicator:', '^opinion:', '^study:', '^poll:', '^special report:', '^report:', '^analysis:', '^alert:', '^fact check:', '^interview:', '^\[salon\]', '^\[forbes\]', '^news wrap:', '^\[national\] -', '^authorities:', '^officials:', '^news brief:', '^watch:', '^watch live:', '^breaking:', '^the latest:', 'https:\/\/url4ever.com(.*)$', '| february 24, 2021$', '\[ap\]$', '\(ap\)$', ': coronavirus updates$', '\.', ',']
    for reg in regs:
        df['title'].replace('(?i)' + reg, '', inplace=True, regex=True)

    df['title'] = df['title'].str.strip()
    df = df[df['title'] != '']
    df['lowercase_title'] = df['title'].str.lower()
    df.drop_duplicates(subset='lowercase_title', inplace=True)

    df = df[df['score'] >= threshold]
    df = df[['created_utc', 'title', 'score']]
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
    df['clickbait'] = clickbait

    return df

# load clickbait sources
clickbait = clean_csv('data/training/clickbait.csv.gz', 1, 5)
clickbait = clickbait[clickbait['title'].str.contains('|', regex=False)]
clickbait['title'] = clickbait['title'].str.split('|').str[0].str.strip()
clickbait = clickbait[clickbait['title'] != '']
clickbait.drop_duplicates(subset='title', inplace=True)
#display(clickbait)

# load non_clickbait sources
apnews = clean_csv('data/training/apnews.csv.gz', 0, 1)
npr = clean_csv('data/training/npr.csv.gz', 0, 1)
pbs = clean_csv('data/training/pbs.csv.gz', 0, 1)
reuters = clean_csv('data/training/reuters.csv.gz', 0, 1)
non_clickbait = pd.concat([apnews, npr, pbs, reuters], ignore_index=True)
#display(non_clickbait)

# ensure balance between clickbait and non_clickbait entries
clickbait = clickbait.sample(min(clickbait.shape[0], non_clickbait.shape[0]), ignore_index=True)
non_clickbait = non_clickbait.sample(min(clickbait.shape[0], non_clickbait.shape[0]), ignore_index=True)

# combine clickbait and non_clickbait datasets
training_cleaned = pd.concat([clickbait, non_clickbait], ignore_index=True)
training_cleaned = training_cleaned.sample(frac=1, ignore_index=True)
#display(training_cleaned)

training_cleaned.to_csv('data/training/training_cleaned.csv.gz', index=False, compression="gzip")
