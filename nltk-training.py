#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import time

#nltk.config_megam('megam.opt')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# # Load Data

# In[2]:


training = pd.read_csv('data/training/training_cleaned.csv.gz')
#display(training)


# # Prepare Data

# In[3]:


stopwords = nltk.corpus.stopwords.words("english")

def filter_stopwords(wordlist):
    filtered = []
    
    for word in wordlist:
        if word not in stopwords:
            filtered.append(word)
    
    return filtered

def lemmatize_words(wordlist):
    lemmatized = []
    
    for word in wordlist:
        lemmatized.append(nltk.stem.wordnet.WordNetLemmatizer().lemmatize(word))
    
    return lemmatized

def prepare_titles(data):
    data['title'] = data['title'].str.lower().str.strip()
    data = data[data['title'] != '']
    data.dropna(subset='title', inplace=True)
            
    data['title'] = data['title'].apply(lambda x: nltk.word_tokenize(x))
    #data['title'] = data['title'].apply(lambda x: filter_stopwords(x)) # lowers accuracy
    data['title'] = data['title'].apply(lambda x: lemmatize_words(x))
    
    return data


# In[4]:


def find_feature(wordlist):
    feature = {}
    
    for x in all_features:
        feature[x] = x in wordlist
        
    return feature

def create_featuresets(data, train, num_features=100):
    #if train:
    #    document = [(row['title'], row['clickbait']) for index, row in data.iterrows()]
    #else:
    #    document = [(row['title']) for index, row in data.iterrows()]
    
    all_words = []

    for index, row in data.iterrows():
        for word in row['title']:
            all_words.append(word)       
    
    if train:
        global all_features
        all_features = list(nltk.FreqDist(all_words))[:num_features]
        
        featuresets = np.array(data[['title', 'clickbait']])
    else:
        featuresets = np.array(data[['title']])
    
    # vectorizing using numpy shaves off about 40% of the processing time for this section
    find_vector = np.vectorize(find_feature) 
    featuresets[:,0] = find_vector(featuresets[:,0])
    
    #if train:
    #    featuresets = [(find_feature(wordlist), category) for (wordlist, category) in document]
    #else:
    #    featuresets = [(find_feature(wordlist)) for (wordlist) in document]
        
    return featuresets


# In[5]:


def prepare_data(data, train, num_feaures=100):
    temp = data.copy()
    
    time0 = time.time()
    temp = prepare_titles(temp)
    print('prepare_titles time: {:.2f}s'.format(time.time() - time0))
    
    # used for error analysis below
    #global temp_datacopy
    #temp_datacopy = data.copy()
    
    time0 = time.time()
    temp = create_featuresets(temp, train, num_feaures)
    print('create_featuresets time: {:.2f}s'.format(time.time() - time0))
    
    return temp


# In[6]:


prepared = prepare_data(training, True, 5)
train_ratio = 0.75
train, test = prepared[:int(len(prepared) * train_ratio)], prepared[int(len(prepared) * train_ratio):]


# # Train Models

# In[7]:


time0 = time.time()
nbclassifier = nltk.NaiveBayesClassifier.train(train)
print('training time: {:.2f}s'.format(time.time() - time0))


# In[8]:


# trains too slowly
#time0 = time.time()
#meclassifier_iis = nltk.MaxentClassifier.train(train, algorithm='iis', max_iter=5)
#print('\ntraining duration: {:.2f}s'.format(time.time() - time0))

# trains too slowly
#time0 = time.time()
#meclassifier_gis = nltk.MaxentClassifier.train(train, algorithm='gis', max_iter=5)
#print('\ntraining duration: {:.2f}s'.format(time.time() - time0))

# much less accurate than NaiveBayesClassifier
#time0 = time.time()
#meclassifier_megam = nltk.MaxentClassifier.train(train, algorithm='megam')
#print('\ntraining duration: {:.2f}s'.format(time.time() - time0))


# # Test Models

# In[9]:


time0 = time.time()
accuracy = nltk.classify.accuracy(nbclassifier, test)
print('accuracy: {:.4f}, testing time: {:.2f}s'.format(accuracy, time.time() - time0))


# In[10]:


# used for error analysis

#errors = []
#index = 0

#for (words, clickbait) in train:
#    prediction = nbclassifier.classify(words)
    
#    if prediction != clickbait:
#        errors.append((clickbait, temp_datacopy['title'].iloc[index], filtered['title'].iloc[index]))

#    index += 1
    
#for e in errors:
#    if e[0] == 1:
#        print('clickbait: yes')
#    else:
#        print('clickbait: no')

#    print(e[2])
#    print(e[3] + '\n')


# In[11]:


# trains too slowly
#time0 = time.time()
#print(nltk.classify.accuracy(meclassifier_iis, test))
#print('\ntesting duration: {:.2f}s'.format(time.time() - time0))

# trains too slowly
#time0 = time.time()
#print(nltk.classify.accuracy(meclassifier_gis, test))
#print('\ntesting duration: {:.2f}s'.format(time.time() - time0))

# much less accurate than NaiveBayesClassifier
#time0 = time.time()
#print(nltk.classify.accuracy(meclassifier_megam, test))
#print('\ntesting duration: {:.2f}s'.format(time.time() - time0))


# # Classify

# In[12]:


files = [('data/classification/nottheonion.csv.gz', 'nottheonion'), 
         ('data/classification/politics.csv.gz', 'politics'), 
         ('data/classification/upliftingnews.csv.gz', 'upliftingnews'), 
         ('data/classification/worldnews.csv.gz', 'worldnews'), 
         ('data/classification/news.csv.gz', 'news'),
         ('data/classification/canadapolitics.csv.gz', 'canadapolitics')]


# In[13]:


dfs = []

for file in files:
    dfs.append((pd.read_csv(file[0]), file[1]))


# In[14]:


filtered_dfs = []

for df_info in dfs:
    df = df_info[0]
    df = df[df['score'] > 50]
    df = df[['created_utc', 'title']]
    df['created_utc'] = pd.to_datetime(df['created_utc'], format='%Y-%m-%d %H:%M:%S')
    filtered_dfs.append((df, df_info[1]))


# In[15]:


featuresets = []

for df_info in filtered_dfs:
    df = df_info[0]
    df = prepare_data(df, False)
    featuresets.append((df, df_info[1]))


# In[16]:


def custom_classify(featureset):
    return nltk.NaiveBayesClassifier.classify(nbclassifier, featureset)


# In[17]:


vector_classify = np.vectorize(custom_classify)


# In[18]:


df_predictions = filtered_dfs.copy()

for i in range(len(featuresets)):
    predictions = np.array(featuresets[i][0])
    predictions = vector_classify(predictions)
    df_predictions[i][0]['nb_predictions'] = predictions
    df_predictions[i][0].to_csv('data/predictions/nltk_predictions_' + featuresets[i][1] + '.csv.gz', index=False, compression="gzip")


# # Visualize Results

# In[19]:


def visualize_results(data, thresh):
    temp = data.copy()
    temp['created_utc'] = temp['created_utc'].dt.year
    
    result_types = ['nb']
    result = pd.DataFrame()
    
    for rt in result_types:
        count = temp.pivot_table(index='created_utc', columns=rt + '_predictions', aggfunc='size')
        count[rt + '_ratio'] = count[1] / (count[1] + count[0])
        count = count[(count[1] + count[0]) >= thresh]
        result = result.join(count[[rt + '_ratio']], how='right')
    
    return result


# In[20]:


plt.figure(figsize=(12,8))

for df_info in df_predictions:
    result = visualize_results(df_info[0], 150)
    plt.plot(result * 100)

plt.title('Percentage of clickbait titles in selected news subreddits (2014-2021)')
plt.legend(['r/NotTheOnion', 'r/politics', 'r/UpliftingNews', 'r/worldnews', 'r/news', 'r/CanadaPolitics'])
plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.savefig('nltk_analysis.png')
plt.close('all')

# In[ ]:




