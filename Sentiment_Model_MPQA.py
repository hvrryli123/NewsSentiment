import nltk
from Noise_Removal import lemmatize_sentence, remove_noise
from Single_Article_Scrape import scrape_news
import pandas as pd

positive_MPQA = pd.read_csv("C:/Users/Admin/OneDrive/Documents/Data Science/News Sentiment/Model_Data/MPQA-OpinionCorpus-NegativeList.csv")
negative_MPQA = pd.read_csv("C:/Users/Admin/OneDrive/Documents/Data Science/News Sentiment/Model_Data/MPQA-OpinionCorpus-PositiveList.csv")
positive_MPQA['Sentiment'] = 'Positive'
negative_MPQA['Sentiment'] = 'Negative'

positive_tokens = positive_MPQA.values.tolist()
negative_tokens = negative_MPQA.values.tolist()

positive_data = dict(positive_tokens)
negative_data = dict(negative_tokens)
dataset = positive_data | negative_data

import random
keys = list(dataset.keys())
random.shuffle(keys)

ShuffledDataset = dict()
for key in keys:
    ShuffledDataset.update({key: dataset[key]})
    
from nltk import classify
from nltk import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(dataset)

custom_tokens = remove_noise(scrape_news("https://www.foxnews.com/politics"))

#print(classifier.classify(dict([token, True] for token in custom_tokens)))


"""
positives = ['Positive' for x in positive_tokens]
negatives = ['Negative' for x in negative_tokens]

positive_data = zip(positive_tokens, positives)
negative_data = zip(negative_tokens, negatives)
print(positive_data)




positive_data = [(x, 'Positive') for x in positive_tokens]
negative_data = [(x, 'Negative') for x in negative_tokens]



import random
random.shuffle(dataset)

print(positive_data)


dataset_tup = tuple(i for i in dataset)
print(dataset_tup)

print(type(dataset_tup))

#dataset_dict = dict((y, x) for x, y in dataset_tup)


dataset_dict = {}
for a, b in dataset_tup:
    dataset_dict.setdefault(a, []).append(b)
    
print(len(dataset_dict))





dataset = pd.concat([positive_MPQA, negative_MPQA])

dataset = dataset.sample(frac=1).reset_index(drop=True)

train_data = dataset.values.tolist()


"""