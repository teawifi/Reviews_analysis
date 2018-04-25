# Train doc2vec model


import pandas as pd
import nltk
import re
#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

def review_to_wordlist(review, remove_stopwords=False, convert_number2word = False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #
    # 2. Convert numbers to words

    # 3. Replace numbers to "NUM"
    review_text = re.sub("(\d+(\.|\,)?\d+)", 'NUM', review_text)
    # 4. Remove URLs
    review_text = re.sub("(https?:\/\/(?:\w+\.|(?!www\.))?\w+\.[^\s]{,10}|www\.\w+\.?[^\s]{,10}|https?:\/\/(?:www\.|(?!www))[\w+]\.[^\s]{,10}|https?:\/\/\w+)", "", review_text)    
    # 5. Remove non-letters
    review_text = re.sub("[^a-zA-Z!]", " ", review_text)
    #
    # 6. Convert words to lower case and split them
    words = word_tokenize(review_text.lower())
    # words = review_text.lower().split()
    #
    # 7. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 8. Return a list of words
    return (words)

if __name__ == '__main__':
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)   

    tagged_documents = []

    for id, review in zip(train.id, train.review):
        tagged_documents.append(TaggedDocument(words=review_to_wordlist(review), tags=[str(id)]))

    for id, review in zip(unlabeled_train.id, unlabeled_train.review):
        tagged_documents.append(TaggedDocument(words=review_to_wordlist(review), tags=[str(id)]))

    for id, review in zip(test.id, test.review):
        tagged_documents.append(TaggedDocument(words=review_to_wordlist(review), tags=[str(id)]))   

    max_epochs = 20
    vec_size = 300    
    min_word_count = 15
    num_workers = 4
    context = 10
    total_examples = len(tagged_documents)
    print("total_examples: ", total_examples)
	
    model = Doc2Vec(documents=tagged_documents,
                    vector_size=vec_size,
                    min_count=min_word_count,
                    dm=1,
                    workers=num_workers,
                    window=context,
                    epochs=max_epochs )
   
    model.train(documents=tagged_documents, total_examples=total_examples, epochs=model.epochs) 

    model.save("pv_dm_300features_15minwords_10context.d2vmodel")
    print("Model saved")