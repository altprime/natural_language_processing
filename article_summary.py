import nltk
#nltk.download("stopwords")
#nltk.download("punkt")
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from string import punctuation
from nltk.stem.lancaster import LancasterStemmer
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import operator

link = None
number = 0

def summarizer(link, number):
    print("This program will summarize any article into a small number of sentences. \n")
    print("Please provide the URL and the number of sentences. \n")
    #print("Enter the requred fields: \n")
    link = input("Enter URL: ")
    number = int(input("Enter number of sentences: "))

    custom = set(stopwords.words("english")+list(punctuation)+["\xa0","'","...","..","''","``","-","--",])

    req = Request(link, headers={'User-Agent': 'Mozilla/5.0'})
    response = urlopen(req)
    page = urlopen(req).read()

    soup = BeautifulSoup(page, "lxml")
    paras = ' '.join([p.text for p in soup.findAll('p')])

    # tokenize
    paras_sentence = sent_tokenize(paras)
    paras_words = word_tokenize(paras)

    # stopwrods removal
    words = [word for word in paras_words if word not in custom]
    words = [word.lower() for word in words]

    # counter
    counts = Counter(words)
    counts = counts.most_common()

    final_words_needed = [x for x in counts if x[1] > 3]
    final_words_needed

    # dictionary
    d = {}
    for s in paras_sentence:
        score = 0
        for w in final_words_needed:
            if w[0] in s:
                score = score + w[1]
        d[s] = score

    score_dict = sorted(d.items(), key = operator.itemgetter(1), reverse = True)
    score_dict

    our_summary = []
    for x in score_dict[:3]:
        our_summary.append(x[0])

    our_summary = ' '.join(our_summary[:])
    return print("\n", our_summary)


summarizer(link, number)
