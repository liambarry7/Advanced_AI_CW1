import pandas as pd
import spacy
import spacy.cli
from nltk.stem import PorterStemmer

# spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')

'''
-	Parsing or tokenisation:
        Breaking a textual sentence into tokens and label them with predefined tags
-	Part-of-Speech (PoS) tagging:
        Identifying the part of speech for every word, or n-words (n-gram)
-	Lemmatization:
        Reducing the various forms of a word into a single form for easy analysis
-	Stemming:
        Cutting the inflected words to their root form
-	Named Entity Recognition (NER):
        Identifying the named entities in a doc e.g. person, organisation and location
'''

def sentence_segmentation(text):
    sentences = []

    doc = nlp(text)
    for sentence in doc.sents:
        sentences.append(sentence.text)

    return sentences

def tokenization(text):
    tokens = []
    sentences = sentence_segmentation(text)
    for sent in sentences:
        doc = nlp(sent.lower().strip()) # format all text to lowercase and remove whitespaces
        for token in doc:
            tokens.append(token.text)

    return ' '.join(tokens)


def part_of_speech(text):
    tokens = []
    doc = nlp(text)
    for token in doc:
        print(
            f"{str(token.text):15}"
            f"{str(token.pos_):8}"
            f"{str(token.tag_):6}"
            f"{str(spacy.explain(token.tag_))}"
        )


def stop_word_removal(text):
    tokens = []
    doc = nlp(text)
    for token in doc:
        print(f"{token.text} : {token.is_stop}")
        if not token.is_stop:
            tokens.append(token.text)

    return ' '.join(tokens)

def remove_punc(text):
    tokens = []
    doc = nlp(text)
    for token in doc:
        print(f"{token.text} : {token.is_punct}")
        if not token.is_punct:
            tokens.append(token.text)

    return ' '.join(tokens)

def lemmatization(text):
    tokens = []
    doc = nlp(text)
    for token in doc:
        print(f"{token.text} : {token.lemma_}")
        tokens.append(token.lemma_)

    return ' '.join(tokens)

def stemming(text):
    tokens = []
    doc = nlp(text)
    ps = PorterStemmer()
    for token in doc:
        print(f"{token.text} : {ps.stem(token.text)}")
        tokens.append(ps.stem(token.text))

    return ' '.join(tokens)

def named_entity_recognition(text):
    token = []
    doc = nlp(text)
    for ent in doc.ents:
        print(
            f"{str(ent.text):20}",
            f"{str(ent.label_):8}",
            f"{str(spacy.explain(ent.label_))}"
        )




def test():
    print("Test harness")

    df = pd.read_csv("social-media-release.csv")
    print(df.head())

    # get example post
    example_post = df.iloc[100]
    print(example_post)

    # get post text
    post_text = example_post['post']
    print(post_text)

    sentences = sentence_segmentation(post_text)
    print(sentences)

    tokens = tokenization(post_text)
    print(tokens)

    tokens = stop_word_removal(tokens)
    print(tokens)

    tokens = remove_punc(tokens)
    print(tokens)

    tokens_lem = lemmatization(tokens)
    print(tokens_lem)

    tokens_stem = stemming(tokens)
    print(tokens_stem)

    pos = part_of_speech(tokens)

    ner = named_entity_recognition(tokens)

if __name__ == "__main__":
    test()