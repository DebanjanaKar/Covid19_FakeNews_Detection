#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:54:19 2020

@author: suranjana
"""





import urllib
import requests, re, spacy
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import numpy as np
import csv
import preprocessor as p
from gensim.parsing.preprocessing import remove_stopwords
from googletrans import Translator

translator = Translator()
ua = UserAgent()
nlp = spacy.load("en_core_web_sm")
f = open('CommonWords.txt')
commonEngWords = f.read().splitlines()
f.close()
popular_links = [
        "nytimes", "wsj", "huffpost", "washingtonpost","time","republicworld",
        "latimes", "reuters", "abcnews", "usatoday",
        "bloomberg", "nbcnews", "dailymail", "theguardian",
        "thesun", "mirror", "telegraph", "bbc",
        "thestar", "theglobeandmail", "forbes",
        "cnbc", "chinadaily", "nypost", "usnews",
        "timesofindia", "thehindu", "hindustantimes",
        "cbsnews", "sfgate", "thehill", "thedailybeast",
        "newsweek", "theatlantic", "nzherald", "vanguardngr",
        "dailysun", "thejakartapost", "thestar",
        "straitstimes", "bangkokpost", "japantimes",
        "thedailystar", "scmp", "yahoo.com/news", "news.google"
        ]


def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])





def get_search_result(query, number_result = 20):
    query = urllib.parse.quote_plus(query) # Format into URL encoding
    
    
    google_url = "https://www.google.com/search?q=" + query + "&num=" + str(number_result)
    response = requests.get(google_url, {"User-Agent": ua.random})
    soup = BeautifulSoup(response.text, "html.parser")
    
    result_div = soup.find_all('div', attrs = {'class': 'ZINbbc'})
    
    links = []
    titles = []
    descriptions = []
    for r in result_div:
        # Checks if each element is present, else, raise exception
        try:
            link = r.find('a', href = True)
            title = r.find('div', attrs={'class':'vvjwJb'}).get_text()
            description = r.find('div', attrs={'class':'s3v9rd'}).get_text()
            
            # Check to make sure everything is present before appending
            if link != '' and title != '' and description != '': 
                links.append(link['href'])
                titles.append(title)
                descriptions.append(description)
        # Next loop if one element is not present
        except:
            continue
    return links, titles, descriptions

 
def clean_links(links, titles, descriptions):
    to_remove = []
    clean_links = []
    for i, l in enumerate(links):
        clean = re.search('\/url\?q\=(.*)\&sa',l)
    
        # Anything that doesn't fit the above pattern will be removed
        if clean is None:
            to_remove.append(i)
            continue
        clean_links.append(clean.group(1))
    
    # Remove the corresponding titles & descriptions
    for x in to_remove:
        del titles[x]
        del descriptions[x]
    return clean_links, titles, descriptions


def filter_links(links, titles, descriptions):
    to_remove = []
    for i, l in enumerate(links):
        if not any(a in l for a in popular_links):
            to_remove.append(i)
            continue
    
    # Remove the corresponding titles & descriptions
    links = [l for i,l in enumerate(links) if i not in to_remove]
    titles = [t for i,t in enumerate(titles) if i not in to_remove]
    links = [l for i,d in enumerate(descriptions) if i not in to_remove]
    return links, titles, descriptions
    

def valid_links(links):
    no_link = 0
    for link in links:
        if any(implink in link for implink in popular_links):
            no_link = no_link+1
    return no_link/len(links)


def split_sentence(text):
    nlp = spacy.blank('en')
    nlp.add_pipe(PySBDFactory(nlp))
    doc = nlp(text)
    return([sent.text for sent in doc.sents if sent.text.isspace()==False])


def valid_description(query, descriptions):
    temp_query = preprocessing(query)
    sentences = split_sentence(temp_query)
    score = []
    for q in query:
        temp = [levenshtein_ratio_and_distance(q, sent,ratio_calc = True) for sent in sentences]
        score.append(max(temp))
    return sum(score)/len(score)


def preprocessing(text):
    p.set_options(p.OPT.URL, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)
    text = p.clean(text)
    text = remove_stopwords(text)
    text = text.lower().replace('[^\w\s]',' ').replace('\s\s+', ' ').replace('@','').replace('#','. ').replace('&amp;', 'and')
    return text


def translate_text(text):
    result = translator.translate('Mitä sinä teet')
    return result.text

    
if __name__ == "__main__":
    query = "'trade war'"
    threshold = 0.3
    tweets = []
    clean_tweets = []
    labels = []
    with open('en_dataset.csv', 'rt') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                tweets.append(row[2])
                clean_tweets.append(preprocessing(row[2]))
                labels.append(row[3])
            line_count += 1
    score = []
    for query in tweets:
        links, titles, descriptions = get_search_result(query)
        links, titles, descriptions = filter_links(links, titles, descriptions)
        #descriptions = [preprocessing(desc) for desc in descriptions]
        if len(titles)>0:
            titles = [preprocessing(title.lower()) for title in titles]
            desc_score = valid_description(query, titles)
            score.append(desc_score)
        else:
            score.append(0)
    
    
