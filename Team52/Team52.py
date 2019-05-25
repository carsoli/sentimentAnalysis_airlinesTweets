
# coding: utf-8

# ## Imports

# In[ ]:


import numpy as np
import pandas as pd 
from collections import Counter
import re, string, math

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer

from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from nltk.corpus import stopwords, words

from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ## Functions

# In[ ]:


def df_split(x, y, train_size=0.8, shuffle=False):
    return train_test_split(x, y, train_size=train_size, test_size=1-train_size, shuffle=shuffle)


# In[ ]:


def create_word_count_map(text):
    word_regex = re.compile(r"[\w']+")
    words = word_regex.findall(text)
    return Counter(words)


# In[ ]:


#compute cosine similarity between vectorized tokens using sklearn
def compute_cosine_similarity(x, vectorizer):
    transformed_x = vectorizer.fit_transform(x)
    return cosine_similarity(transformed_x)


# In[ ]:


#https://stackoverflow.com/questions/15173225/calculate-cosine-similarity-given-2-sentence-strings
def compute_cosine_similarity2(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection]) #dot product of the intersection set

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator


# In[ ]:


def filter_retweet(tweet):
    if not(re.compile(r'RT @')).search(tweet):
        return tweet


# In[ ]:


def filter_short_tweet(tweet):
    if not(len(tweet) < 20):
        return tweet 


# In[ ]:


english_vocab_set = set(words.words())
def filter_nonenglish_tweet(tweet):
    tweet_count_vec = create_word_count_map(tweet)
    tweet_word_set = set(tweet_count_vec.keys())
    #sum of all values of count vec
    tweet_wordcount = sum(tweet_count_vec[w] for w in tweet_count_vec.keys() ) 
    
    non_english_set = tweet_word_set - english_vocab_set
    non_english_wordcount = sum(tweet_count_vec[fw] for fw in non_english_set) 
   
    if not( (100 - (non_english_wordcount/tweet_wordcount)*100) < 15):
        return tweet


# In[ ]:


def append_df_elem(df, columns, data_arr, idx=0, ignore_index=True):
    df = df.append(pd.DataFrame(data=[data_arr[idx]], columns = columns), 
                                     ignore_index=ignore_index)
    return df


# In[ ]:


def filter_similar_tweets(df, vectorizer): #df is primarily filtered       
    x = df['text']
    cos_matrix = compute_cosine_similarity(x, vectorizer)
    rows = cos_matrix.shape[0]
    rep_pairs = [[]] #debugging
    repeated = []

    #no need to iterate over the entire array
    for r in range(0, rows-2): 
        for c in range(r+1, rows-1):
            if cos_matrix[r][c]>=0.9:
                rep_pairs.append([r,c])
                if c in repeated:
                    continue
                else: 
                    if r in repeated:
                        repeated.append(c)
                    else:
                        repeated.append(r)
    #for n similar tweets, only n-1 are added to the repeated; no priorities                
    return df.drop(repeated)


# In[ ]:


def filter_similar_tweets2(df): #df is primarily filtered
    tweets = df['text']
    df_columns = df.columns
    df_filtered = pd.DataFrame(columns = df_columns)
    df_data = np.array(df)
    df_filtered = append_df_elem(df_filtered, df_columns, df_data, 0)
    
    vec1 = Counter()
    vec2 = [Counter()]
    vec2[0] = create_word_count_map(np.array(df_filtered['text'])[0])
    
    for idx, ft in enumerate(tweets):
        vec1 = create_word_count_map(ft)
        for idx2, fft in enumerate(df_filtered):
            if compute_cosine_similarity2(vec1, vec2[idx2])>=0.9:
                break
            else:
                df_filtered = append_df_elem(df_filtered, df_columns, df_data, idx)
                vec2.append(vec1)
                
    return df_filtered #final data frame      


# In[ ]:


def filter_dataframe(df, vectorizer):
    tweets = df['text']
    df_columns = df.columns
    df_filtered = pd.DataFrame(columns = df_columns)
    df_data = np.array(df)
    
    for idx, tweet in enumerate(tweets):
        if not(filter_short_tweet(tweet) == None) and not(filter_retweet(tweet) == None) and not(filter_nonenglish_tweet(tweet) == None):
            df_filtered = append_df_elem(df_filtered, df_columns, df_data, idx)     
    
    #df_filtered = filter_similar_tweets2(df_filtered) 
    df_filtered = filter_similar_tweets(df, vectorizer)
    return df_filtered


# In[ ]:


#lancaster_stemmer = LancasterStemmer()
def preprocess_and_tokenize(tweet):
    tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True) 
    regex_word_tokenizer = RegexpTokenizer(r'\w+')

    porter_stemmer = PorterStemmer()
    wnl = WordNetLemmatizer()

    stop_words = stopwords.words('english')
    stop_words += list(string.punctuation)
    stop_words += ['``', "''", "'s"]
    
    tokens = set()
    
    t = tweet
    t = t.casefold() 
    #remove urls
    t = re.sub(r'https?:\/\/\S+\s*', '', t) 
    t = re.sub(r'www\.\S+\s*', '', t) 
    #remove numbers, remove numbers followed by letters
    t = re.sub(r'[0-9](\S*)', '', t)
    #tweet_tokens; a list
    t_tokens = tweet_tokenizer.tokenize(t)
    t = ' '.join(t_tokens)

    t_tokens = regex_word_tokenizer.tokenize(t) #tokenize tweet with regex
    t_tokens = [tt for tt in t_tokens if not tt in stop_words] #remove stop words & punctuation
    t_tokens = [wnl.lemmatize(tt) for tt in t_tokens] #lemmatize all tokens
    t_tokens = [porter_stemmer.stem(tt) for tt in t_tokens] #stem all tokens

    for tt in t_tokens:    
        tokens.add(tt) #tt: one tweet token, in t_tokens of a tweet
        
    return tokens


# ## Sentiment Analysis: Twitter_Airline

# In[ ]:


#reads ds 
df = pd.read_csv('~/.kaggle/twitter-airline-sentiment/Tweets.csv')
#print(df)


# ### Original Dataframe

# In[ ]:


#original data set
x = np.array(df['text'])
y = np.array(df['airline_sentiment'])
x_train, x_test, y_train, y_test = df_split(x, y, train_size=0.8, shuffle=True)


# In[ ]:


#Count Vectorizer; constructs a document-word count matrix
cv = CountVectorizer(tokenizer= lambda t: list(preprocess_and_tokenize(t)), strip_accents='ascii')
transformed_train_cv = cv.fit_transform(x_train)
print(transformed_train_cv.shape)


# In[ ]:


#Tf-Idf Vectorizer; constructs a tf-idf matrix
tidv = TfidfVectorizer(tokenizer= lambda t: list(preprocess_and_tokenize(t)), strip_accents='ascii')
transformed_train_tfidf = tidv.fit_transform(x_train)
print(transformed_train_tfidf.shape)


# In[ ]:


#transform the test inputs using cv and tfidf(no fitting is done on test)
transformed_test_cv = cv.transform(x_test)

transformed_test_tfidf = tidv.transform(x_test)


# In[ ]:


#Multi-nomial NB classifier:
#using count vectorizer: feed the classifier train inputs, and labels
mnb_classifier1 = MultinomialNB(alpha=1)
mnb_classifier1.fit(transformed_train_cv, y_train)
#using tfidf vectorizer
mnb_classifier2 = MultinomialNB(alpha=1)
mnb_classifier2.fit(transformed_train_tfidf, y_train)


# In[ ]:


#use the trained classifiers to predict the output(sentiment_class)
#compare the prediction with the labels of the train data
#compute accuracy of classifier
predictions_mnb1 = mnb_classifier1.predict(transformed_test_cv)
print("f1_score for MNB classifier with CountVectorizer %r" 
      %(100*metrics.f1_score(y_test, predictions_mnb1, average='micro')))

#repeat for tfidfv
predictions_mnb2 = mnb_classifier2.predict(transformed_test_tfidf)

print("f1_score for MNB classifier with Tf-Idf Vectorizer %r" 
      %(100*metrics.f1_score(y_test, predictions_mnb2, average='micro')))


# In[ ]:


#K-Nearest Neighbour Classifier:
n = 1 # test value
knn_classifier1 = KNeighborsClassifier(n_neighbors= n, weights='distance')
knn_classifier1.fit(transformed_train_cv, y_train)

predictions_knn1 = knn_classifier1.predict(transformed_test_cv)
print("f1_score for KNN classifier with CountVectorizer %r" 
      %(100*metrics.f1_score(y_test, predictions_knn1, average='micro')))

knn_classifier2 = KNeighborsClassifier(n_neighbors= n, weights='distance')
knn_classifier2.fit(transformed_train_tfidf, y_train)

predictions_knn2 = knn_classifier2.predict(transformed_test_tfidf)
print("f1_score for KNN classifier with Tf-Idf Vectorizer %r"
      %(100*metrics.f1_score(y_test, predictions_knn2, average='micro')))


# In[ ]:


#Random Forest Classifier
rf_classifier1 = RandomForestClassifier(random_state = 0)
rf_classifier1.fit(transformed_train_cv, y_train)

predictions_rf1 = rf_classifier1.predict(transformed_test_cv)
print("f1_score for RF classifier with CountVectorizer %r"
      %(100*metrics.f1_score(y_test,predictions_rf1, average='micro')))

rf_classifier2 = RandomForestClassifier(random_state = 0)
rf_classifier2.fit(transformed_train_tfidf, y_train)

predictions_rf2 = rf_classifier2.predict(transformed_test_tfidf)
print("f1_score for RF classifier with Tf-Idf Vectorizer %r"
    %(100*metrics.f1_score(y_test,predictions_rf2, average='micro')))


# ### Filtered Dataframe (Bonus)

# In[ ]:


#filtered data set
vectorizer = tidv = TfidfVectorizer(tokenizer= lambda t: list(preprocess_and_tokenize(t)), strip_accents='ascii')
df_filtered = filter_dataframe(df, vectorizer)

x_filtered = df_filtered['text']
y_filtered = df_filtered['airline_sentiment'] 
x_train_filtered, x_test_filtered, y_train_filtered, y_test_filtered = df_split(
    x_filtered, y_filtered, train_size=0.8, shuffle=True)


# In[ ]:


cv_filtered = CountVectorizer(tokenizer= lambda t: list(preprocess_and_tokenize(t)), strip_accents='ascii')
transformed_train_cv_filtered = cv_filtered.fit_transform(x_train_filtered)
print(transformed_train_cv_filtered.shape)

tidv_filtered = TfidfVectorizer(tokenizer= lambda t: list(preprocess_and_tokenize(t)), strip_accents='ascii')
transformed_train_tfidf_filtered = tidv_filtered.fit_transform(x_train_filtered)
print(transformed_train_tfidf_filtered.shape)

transformed_test_cv_filtered = cv_filtered.transform(x_test_filtered)
transformed_test_tfidf_filtered = tidv_filtered.transform(x_test_filtered)


# In[ ]:


fmnb_classifier1 = MultinomialNB(alpha=1)
fmnb_classifier1.fit(transformed_train_cv_filtered, y_train_filtered)

fmnb_classifier2 = MultinomialNB(alpha=1)
fmnb_classifier2.fit(transformed_train_tfidf_filtered, y_train_filtered)


# In[ ]:


fpredictions_mnb1 = fmnb_classifier1.predict(transformed_test_cv_filtered)
print("f1_score for MNB classifier with CountVectorizer %r" 
      %(100*metrics.f1_score(y_test_filtered, fpredictions_mnb1, average='micro')))

fpredictions_mnb2 = fmnb_classifier2.predict(transformed_test_tfidf_filtered)

print("f1_score for MNB classifier with Tf-Idf Vectorizer %r" 
      %(100*metrics.f1_score(y_test_filtered, fpredictions_mnb2, average='micro')))


# In[ ]:


#K-Nearest Neighbour Classifier:
n = 2 
fknn_classifier1 = KNeighborsClassifier(n_neighbors=n, weights='distance')
fknn_classifier1.fit(transformed_train_cv_filtered, y_train_filtered)

fpredictions_knn1 = fknn_classifier1.predict(transformed_test_cv_filtered)
print("f1_score for KNN classifier with CountVectorizer %r" 
      %(100*metrics.f1_score(y_test_filtered, fpredictions_knn1, average='micro')))

fknn_classifier2 = KNeighborsClassifier(n_neighbors= n, weights='distance')
fknn_classifier2.fit(transformed_train_tfidf_filtered, y_train_filtered)

fpredictions_knn2 = fknn_classifier2.predict(transformed_test_tfidf_filtered)
print("f1_score for KNN classifier with Tf-Idf Vectorizer %r"
      %(100*metrics.f1_score(y_test_filtered, fpredictions_knn2, average='micro')))


# In[ ]:


frf_classifier1 = RandomForestClassifier(random_state = 0)
frf_classifier1.fit(transformed_train_cv_filtered, y_train_filtered)

fpredictions_rf1 = frf_classifier1.predict(transformed_test_cv_filtered)
print("f1_score for RF classifier with Count Vectorizer %r"
    %(100*metrics.f1_score(y_test_filtered,fpredictions_rf1, average='micro')))

frf_classifier2 = RandomForestClassifier(random_state = 0)
frf_classifier2.fit(transformed_train_cv_filtered, y_train_filtered)

fpredictions_rf2 = frf_classifier2.predict(transformed_test_tfidf_filtered)
print("f1_score for RF classifier with Tf-Idf Vectorizer %r"
    %(100*metrics.f1_score(y_test_filtered,fpredictions_rf2, average='micro')) )

