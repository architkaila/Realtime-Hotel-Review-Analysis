## Importing the libraries

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import spacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import texthero as hero
import re
import os
from tqdm import tqdm
import string
import openai
import stanza

## Load the spacy model
nlp = spacy.load("en_core_web_sm", enable=["tokenizer", "lemmatizer"])

## Load the stanza model
nlp_stanza = stanza.Pipeline("en")

def read_data(datafile):
    """
    Read the data from the csv file

    Args:
        datafile (str): path to the csv file
    
    Returns:
        data (pd.DataFrame): dataframe containing the data
    """
    data = pd.read_csv(datafile)
    
    return data

def rating_to_sentiment(rating):
    """
    Convert the rating to sentiment (0: negative, 1: positive)

    Args:
        rating (int): rating of the review
    
    Returns:
        sentiment (int): sentiment of the review
    """
    if rating>4:
        ## Positive Sentiment
        return 1
    elif rating < 2:
        ## Negative Sentiment
        return 0

def tokenize_and_lemmatize(sentence, nlp):
    """
    Tokenize and Lemmatize the sentence

    Args:
        sentence (str): sentence to be tokenized
        nlp (spacy): spacy model
    
    Returns:
        processed_sentence (str): tokenized and lematized sentence
    """

    ## get the stopwords and punctuations
    punctuations = string.punctuation
    stopwords = list(STOP_WORDS)

    ## Tokenize the sentence using spacy
    tokens = nlp(sentence)

    ## Lemmatize the tokens
    tokens = [word.lemma_.lower().strip() for word in tokens]

    ## Create the sentence back from the tokens
    processed_sentence = " ".join([i for i in tokens])
    
    return processed_sentence

def clean_text(data, col_name):
    """
    Clean a text column of the dataframe using texthero

    Args:
        data (pd.DataFrame): dataframe containing the data
        col_name (str): name of the column to be cleaned
    
    Returns:
        data (pd.DataFrame): dataframe containing the cleaned data
        clean_col_name (str): name of the cleaned column
    """

    ## Clean the text column using texthero
    clean_col_name = f'cleaned_{col_name}'
    data[clean_col_name] = data[col_name].pipe(hero.clean)
    
    return data, clean_col_name

def run_pipiline(data):
    """
    Run the data cleaning pipeline

    Args:
        data (pd.DataFrame): dataframe containing the data
    
    Returns:
        data_cleaned (pd.DataFrame): dataframe containing the cleaned data
    """
    
    ## Print the value counts of the rating
    print("[INFO] RAW data value counts: \n", data.Rating.value_counts(), "\n")

    ## Convert the rating to sentiment
    print("[INFO] Converting the rating to sentiment...")
    data['Sentiment'] = data['Rating'].apply(rating_to_sentiment)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    ## Print the value counts of the rating
    print("[INFO] Sentiment data value counts: \n", data.Sentiment.value_counts(), "\n")

    ## Clean data and remove unwanted characters
    print("[INFO] Cleaning the data: removing unwanted characters...")
    data_cleaned, cleaned_col_name = clean_text(data.copy(), col_name='Review')

    ## Tokenize and Lemmatize the data
    print("[INFO] Tokenizing and Lemmatizing the data...")
    tqdm.pandas()
    data_cleaned[cleaned_col_name] = data_cleaned[cleaned_col_name].progress_apply(lambda x: tokenize_and_lemmatize(x, nlp))

    return data_cleaned
    
if __name__ == "__main__":

    print("[INFO] Reading the data...")
    data = read_data("./data/tripadvisor_hotel_reviews.csv")

    ## Run the pipeline
    data_cleaned = run_pipiline(data=data)

    ## Save the data
    print("[INFO] Saving the data...")
    data_cleaned.to_pickle("./data/cleaned_data.pkl")
    print("[INFO] Data saved successfully!")
