#Web Scraping Single Article
import requests
import pandas as pd
import numpy as np
import matplotlib
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt

def scrape_news(source):
    res = requests.get(source)
    soup = BeautifulSoup(res.content, 'html.parser')

    headlines = soup.find_all('h2',{'class':'title'})

    num_it = len(headlines)
    heading_list = []
    list_links = []
    for i in range(num_it):
        heading_list.append(headlines[i].get_text())
        list_links.append("https://www.foxnews.com" + headlines[i].find('a')['href'])

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
        
    return news_contents
