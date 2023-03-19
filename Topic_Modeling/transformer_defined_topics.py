# Library Imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Local Imports
from config import transformer_to_use
from config import datafile
from config import topic_list

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
    model = SentenceTransformer(transformer_to_use)
    
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

def run_pipeline(data):
    """
    This function takes in a dataframe and returns the updated dataframe with the topics 
    added as a column.

    Args:
        data (dataframe): A dataframe with a column named Review
    
    Returns:
        data (dataframe): A dataframe with an added column named Topics
    """

    # Get the topics for each review
    topics, all_scores = model_topics(data.Review.values.tolist(), topic_list, num_topics=3)
    # Add the topics to the dataframe
    data["Topics"] = topics

    # Get the original reviews
    reviews = data.Review.values.tolist()

    # Print the results for the first 10 reviews
    for i, keywords in enumerate(topics[:10]):
        print('Review {}:\n {}'.format(i, reviews[i]))
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
    updated_data.to_csv("./results/topic_modeling_predefined_topics.csv", index=False)