import nltk
nltk.download('omw-1.4')
nltk.download("twitter_samples")
nltk.download('wordnet')
from Noise_Removal import lemmatize_sentence, remove_noise, get_tweets_for_model
from Single_Article_Scrape import scrape_news
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from nltk.corpus import twitter_samples
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    
positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

import random

positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset



random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]

print(train_data)

from nltk import classify
from nltk import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_data)

custom_tokens = remove_noise(scrape_news("https://www.foxnews.com/politics"))

print(classifier.classify(dict([token, True] for token in custom_tokens)))

