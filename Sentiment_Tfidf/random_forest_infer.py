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
import pickle

from scripts.explanability import explain_prediction
from scripts.grammar_correction import grammar_correction
from scripts.dependency_parser import dependency_parser

from config import explain_sample

## Load the spacy model
nlp = spacy.load("en_core_web_sm", enable=["tokenizer", "lemmatizer"])

## Load the stanza model
nlp_stanza = stanza.Pipeline("en")


def get_explanabiltiy(clf, X_test_df, X_test_vec, y_test, predictions, pred_prob, vec, instance=0):
    
    print("[INFO] Getting the explanation for the instance...")

    ## Get the original review
    print("[INFO] Actual Review: \n", X_test_df.Review[instance], "\n")

    ## Print sentiment classificaiton results
    print("[INFO] Actual Sentiment: ", y_test[instance])
    print("[INFO] Predicted Sentiment: ", predictions[instance])
    print("[INFO] Positive Sentiment Score: ", pred_prob[instance][1])
    print("[INFO] Negative Sentiment Score: ", pred_prob[instance][0])

    ## Get the explanation for the instance
    pred_class = int(predictions[instance])
    force_plot, shap_values = explain_prediction(clf, X_test_vec, pred_class, vec, instance=instance)
    
    ## Feature index giving max shap value
    index = shap_values[pred_class].argmax()

    ## Top adjective giving the max shap value
    candidates = list(vec.get_feature_names_out())
    top_keywords = [candidates[index] for index in shap_values[pred_class].argsort()[0][-4:]]
    top_keywords.reverse()
    top_adjective = candidates[index]
    print("[INFO] Top Keywords: ", top_keywords)
    print("[INFO] Most Important Adjective: ", top_adjective)
    
    ## Get the grammar corrected review
    review_corrected_text = grammar_correction(X_test_df.Review[instance])
    print("[INFO] Grammar Corrected Review: \n", review_corrected_text, "\n")

    ## Get the dependency parsing for the corrected review
    top_adjective_shap, remaining_adjectives = dependency_parser(review_corrected_text, top_keywords, nlp_stanza)

    print("\n")
    print("Top Adjectives and Nouns using Shap Values: \n")
    for adj, noun in top_adjective_shap:
        print(f"{adj} -> {noun}")

    print("\nOther Adjectives and Nouns using Dependency Parsing: \n")
    for adj, noun in remaining_adjectives:
        print(f"{adj} -> {noun}")
    
    return force_plot, shap_values, top_keywords, top_adjective, review_corrected_text, top_adjective_shap, remaining_adjectives, pred_class


def run_pipiline(instance=0):
    """
    Run the pipeline for training the model, saving the model and getting the explanation of an instance from test data.

    Args:
        data (pd.DataFrame): data to train the model
    
    Returns:
        
    """

    ## load the model from disk
    loaded_model = pickle.load(open('./models/final_rf_model.sav', 'rb'))

    ## Load the test data
    X_test_df = pd.read_pickle("./data/X_test.pkl")

    ## Load the vectorizer and transform the test data
    vec = pickle.load(open('./models/vectorizer.sav', 'rb'))
    X_test_vec = vec.transform(X_test_df.Review)

    ## Load the test labels
    with open('./data/y_test.pkl','rb') as f: 
        y_test = pickle.load(f)

    ## Predict on the test data and print the accuracy
    predictions = loaded_model.predict(X_test_vec)
    pred_prob = loaded_model.predict_proba(X_test_vec)

    ## Get the explanation for a instance
    force_plot, shap_values, top_keywords, top_adjective, review_corrected_text, top_adjective_shap, remaining_adjectives, pred_class = get_explanabiltiy(loaded_model, X_test_df, X_test_vec, y_test, predictions, pred_prob, vec, instance=instance)
    
    return force_plot, shap_values, top_keywords, top_adjective, review_corrected_text, top_adjective_shap, remaining_adjectives, pred_class

if __name__ == "__main__":
    print("[INFO] Running the inference pipeline...")
    
    ## Run the pipeline
    force_plot, shap_values, top_keywords, top_adjective, review_corrected_text, top_adjective_shap, remaining_adjectives, pred_class = run_pipiline(instance=explain_sample)