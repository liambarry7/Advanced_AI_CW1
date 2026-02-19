import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # BoW, tfidf

from syntactic_analysis import *

"""
    - TFIDF
    - BoW
    - One-hot encoding
    - LDA
    - Word embedding
    - N-grams?
"""

def bag_of_words(cleaned_text):
    vect = CountVectorizer()
    corpus_BoW = vect.fit_transform([cleaned_text]) # needs text in form of an array
    print(corpus_BoW.toarray())  # print the BoW as array

def tfidf(cleaned_text):
    tfidf_vect = TfidfVectorizer()
    corpus_tfidf = tfidf_vect.fit_transform([cleaned_text])
    print(corpus_tfidf.toarray())

def test():
    print("Test harness")

    df = pd.read_csv("social-media-release.csv")
    print(df.head())

    # get example post
    example_post = df.iloc[14]
    post_text = example_post['post']


    tokens = tokenization(post_text)

    tokens = remove_punc(stop_word_removal(tokens))
    print(tokens)

    tokens = lemmatization(tokens)

    bag_of_words(tokens)
    tfidf(tokens)

    # bag_of_words(post_text)
    # tfidf(post_text)

def one_hot_encoding():
    pass


if __name__ == '__main__':
    test()