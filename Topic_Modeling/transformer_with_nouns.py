# Library Imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import spacy

# Local Imports
from config import transformer_to_use
from config import datafile
from config import topic_list
from config import n_gram_range
from config import transformer_to_use_with_nouns

def model_topics(documents, candidates, num_topics=1):
    """
    This function takes in a list of documents and a list of candidate topics and returns 
    the top num_topics for each document.

    Args:
        documents (list): A list of documents
        candidates (list): A list of candidate topics
        num_topics (int): Number of topics to return for each document
    
    Returns:
        topics (list): A list of top topics for each document
        all_scores (list): A list of scores for each document and candidate topic
    """
    
    # Load the transformer model
    model = SentenceTransformer(transformer_to_use_with_nouns)
    
    # Encode each of the reviews
    doc_embeddings = [model.encode([doc]) for doc in documents]
    
    # Encode the candidate topics
    candidate_embeddings = model.encode(candidates)

    # Calculate cosine similarity between each document and candidate topics
    # Take the top num_topic candidate topics as topics for each document
    all_scores = []
    topics = []
    for doc in doc_embeddings:
        scores = cosine_similarity(doc, candidate_embeddings)
        topic = [candidates[index] for index in scores.argsort()[0][-num_topics:]]
        topics.append(topic)
        all_scores.append(scores)
    
    return topics, all_scores
def find_candidate_topics(review_list, n_gram_range=(1, 1)):
    """
    This function takes in a list of documents and returns a list of candidate topics (nouns).

    Args:
        review_list (list): A list of documents
        n_gram_range (tuple): A tuple of integers indicating the range of n-grams to consider
    
    Returns:
        candidates (list): A list of candidate topics
    """

    # Extract candidate 1-grams and 2-grams 
    vectorizer = CountVectorizer(ngram_range=n_gram_range, stop_words=stopwords.words('english'))
    vectorizer.fit(review_list)
    # Get the feature names
    candidates = vectorizer.get_feature_names_out()

    # Get noun phrases and nouns from articles
    nlp = spacy.load('en_core_web_sm')
    all_nouns = set()
    for doc in review_list:
        doc_processed = nlp(doc)
        # Add noun chunks
        all_nouns.add(chunk.text.strip().lower() for chunk in doc_processed.noun_chunks)
        # Add nouns
        for token in doc_processed:
                if token.pos_ == "NOUN":
                    all_nouns.add(token.text)

    # Filter candidate topics to only those in the nouns set
    candidates = [c for c in candidates if c in all_nouns]

    return candidates

def run_pipeline(data):
    """
    This function takes in a dataframe and returns the updated dataframe with the topics 
    added as a column.

    Args:
        data (dataframe): A dataframe with a column named Review
    
    Returns:
        data (dataframe): A dataframe with an added column named Topics
    """

    review_list = data.Review.values.tolist()
    candidate_topics = find_candidate_topics(review_list, n_gram_range=n_gram_range)

    # Get the topics for each review
    topics, all_scores = model_topics(review_list, candidate_topics, num_topics=5)
    # Add the topics to the dataframe
    data["Topics"] = topics

    # Print the results for the first 10 reviews
    for i, keywords in enumerate(topics[:10]):
        print('Review {}:\n {}'.format(i, review_list[i]))
        print()
        print('Topics: {}'.format(topic_list))
        print('Topic Scores: {}'.format(all_scores[i]))
        print('Final Topic: {}'.format(keywords))
        print()
    
    return data


if __name__ == "__main__":
    
    # Load the data
    data = pd.read_pickle(datafile)

    # Run the pipeline
    updated_data = run_pipeline(data)

    # Save the updated data
    updated_data.to_csv("./results/topic_modeling_from_transformer_nouns.csv", index=False)