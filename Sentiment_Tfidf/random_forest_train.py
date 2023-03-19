## Importing the libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import spacy
import stanza
import pickle

## Import local modules
from explanability import explain_prediction
from Grammar_Correction.grammar_correction import grammar_correction
from Dependency_Parsing.dependency_parser import dependency_parser

## Import params from config
from config import n_gram_range_tfidf
from config import class_weights
from config import explain_sample

## Load the spacy model
nlp = spacy.load("en_core_web_sm", enable=["tokenizer", "lemmatizer"])

## Load the stanza model
nlp_stanza = stanza.Pipeline("en")

def build_features(train_data, test_data, ngram_range, col_name, method='count'):
    """
    Build the numerical features for the model

    Args:
        train_data (pd.DataFrame): training data
        test_data (pd.DataFrame): test data
        ngram_range (tuple): ngram range
        col_name (str): column name containg data to vectorize
        method (str): method to vectorize the data
    
    Returns:
        X_train (pd.DataFrame): featurized training data
        X_test (pd.DataFrame): featurized test data
    """

    if method == 'tfidf':
        ## Create features using TFIDF
        vec = TfidfVectorizer(ngram_range=ngram_range, min_df=800)
        X_train = vec.fit_transform(train_data[col_name])
        X_test = vec.transform(test_data[col_name])

    else:
        ## Create features using word counts
        vec = CountVectorizer(ngram_range=ngram_range, min_df=800)
        X_train = vec.fit_transform(train_data[col_name])
        X_test = vec.transform(test_data[col_name])

    return X_train, X_test, vec

def get_explanabiltiy(clf, X_test_df, X_test_vec, y_test, predictions, pred_prob, vec, instance=1):
    
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


def run_pipiline(data, n_gram_range=(1,1)):
    """
    Run the pipeline for training the model, saving the model and getting the explanation of an instance from test data.

    Args:
        data (pd.DataFrame): data to train the model
        n_gram_range (tuple): ngram range
    
    Returns:
        None
    """

    ## Split the data into train and test
    print("[INFO] Splitting the data into train and test...")
    X_train_df, X_test_df, y_train, y_test = train_test_split(data.drop(columns=["Sentiment", "Rating"]), data.Sentiment.values, test_size=0.2, random_state=0, stratify=data.Sentiment.values)
    
    ## Reset the index for the dataframes
    print("[INFO] Resetting the index for the dataframes...")
    X_train_df.reset_index(drop=True, inplace=True)
    X_test_df.reset_index(drop=True, inplace=True)

    ## Create the numerical features
    print("[INFO] Creating the numerical features...")
    X_train_vec, X_test_vec, vec = build_features(X_train_df, X_test_df, n_gram_range, col_name="cleaned_Review", method='count')

    print(f"[INFO] Train Shape: {X_train_vec.shape} Test Shape: {X_test_vec.shape}")

    ## Train the Random forest model
    clf = RandomForestClassifier(class_weight=class_weights)
    clf.fit(X_train_vec, y_train)

    ## Predict on the test data and print the accuracy
    predictions = clf.predict(X_test_vec)
    pred_prob = clf.predict_proba(X_test_vec)
    print("[INFO] Accuracy: ", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    ## save the model to disk
    pickle.dump(clf, open('./models/final_rf_model.sav', 'wb'))

    ## save the vectorizer to disk
    pickle.dump(vec, open('./models/vectorizer.sav', 'wb'))

    ## Save the test data
    X_test_df.to_pickle("./data/X_test.pkl")

    ## save the test labels
    with open('./data/y_test.pkl','wb') as f: 
        pickle.dump(y_test, f)

    ## Get the explanation for a instance
    get_explanabiltiy(clf, X_test_df, X_test_vec, y_test, predictions, pred_prob, vec, instance=explain_sample)
    
if __name__ == "__main__":

    print("[INFO] Reading the cleaned data...")
    data = pd.read_pickle("./data/clean_lemmatized_data.pkl")

    ## Run the pipeline
    run_pipiline(data=data, n_gram_range=n_gram_range_tfidf)