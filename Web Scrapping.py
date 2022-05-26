#Web Scraping 
import requests
import pandas as pd
import numpy as np
import matplotlib
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt

#from datetime import date
#today = date.today()
#d = today.strftime("%m-%d-%y")
#print("date =" ,d)
#date = 5-21-22

fox_news = "https://www.foxnews.com/politics"
res = requests.get(fox_news)
soup = BeautifulSoup(res.content, 'html.parser')

headlines = soup.find_all('h2',{'class':'title'})

num_it = len(headlines)
heading_list = []
list_links = []
for i in range(num_it):
    heading_list.append(headlines[i].get_text())
    list_links.append("https://www.foxnews.com"+ headlines[i].find('a')['href'])

#list_links = [fox_news + tag.find('a').get('href','') if tag.find('a') else '' for tag in headlines]


news_contents = []
for i in range(num_it):
    link = list_links[i]
    article = requests.get(link)
    article_content = article.content
    soup_article = BeautifulSoup(article_content, 'html.parser')
    body = soup_article.find_all('div', class_='page-content')
    x = body[0].find_all('p')
    list_paragraphs = []
    for p in np.arange(0, len(x)):
        paragraph = x[p].get_text()
        list_paragraphs.append(paragraph)
        final_article = " ".join(list_paragraphs)
        
    news_contents.append(final_article)

#data_fox = pd.DataFrame({"Heading": heading_list, 'Link': list_links, 'Content': news_contents})
#data_fox.to_csv("C:/Users/Admin/OneDrive/Documents/Data Science/News Sentiment/fox_news.csv", index= True)

#NLP
import nltk
nltk.download("punkt")

from nltk.tokenize import sent_tokenize
sent_tk = sent_tokenize(news_contents[0])

#from nltk.tokenize import word_tokenize
#word_tk = word_tokenize(news_contents[0])

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
word_tk = tokenizer.tokenize(news_contents[0])

nltk.download('stopwords')
from nltk.corpus import stopwords
sw = set(stopwords.words('English'))

filtered_text = []
for w in word_tk:
    if w not in sw:
        filtered_text.append(w)

        
#speech tagging
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
pos_tagged_words = pos_tag(word_tk)

#Frequency Distribution
from nltk.probability import FreqDist
fd = FreqDist(filtered_text)

fd.plot(30, cumulative = False)
plt.show()