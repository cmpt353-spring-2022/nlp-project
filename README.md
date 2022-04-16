# nlp-project

## Table of Contents

- [Info](#info)
- [Installation](#installation)
- [Usage](#usage)

## Info

Using the Pushshift Reddit API, we collected clickbait and non-clickbait submission data for training NLP models to differentiate between the two. Clickbait data comes from the subreddit `r/SavedYouAClick` and non-clickbait data comes from the domains `apnews.com`, `npr.org`, `pbs.org`, and `reuters.com`.

After cleaning the data, we used it to train NLP models with 3 different approaches: one used Scikit-learn, one used spaCy, and one used NLTK.

Next, we collected additional submission data from the subreddits `r/CanadaPolitics`, `r/news`, `r/NotTheOnion`, `r/politics`, `r/UpliftingNews`, and `r/worldnews`, which are some of the most popular news subreddits on Reddit. We used our trained NLP models to classify these submissions and then compared the results for each subreddit over time.

> Note: To avoid data leakage, submissions from these 6 subreddits were removed from the data used for training/testing.

Results from all 3 approaches differed a bit, but still showed the same trends, as well as a particularly large spike in clickbait for `r/politics` in 2016. Our report goes into further detail on our methods and findings.

## Installation

### Clone this repo
- https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository

### Download and install python
- https://wiki.python.org/moin/BeginnersGuide/Download

### Download and install pip
- https://pip.pypa.io/en/stable/installation/

### Create virtual environment using virtualenv (Optional)
- Download and install:

    `pip install virtualenv`

- Create environment (Windows):

    `python -m venv venv`

    `./venv/scripts/activate`

- Create environment (Linux/Mac):

    `python -m venv venv`

    `source ./venv/bin/activate`

### Download and install requirements
- `pip install -r requirements.txt`

## Usage

### Data collection

- TODO

### Data cleaning

- TODO

### Model training

- TODO

### Data analysis

- TODO
