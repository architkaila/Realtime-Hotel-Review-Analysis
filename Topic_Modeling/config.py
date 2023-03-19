# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 
                   'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see',
                   'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 
                   'right', 'line', 'even', 'also', 'may', 'take', 'come'])

num_lda_topics = 4
passes=10
chunksize = 10
iterations = 100

transformer_to_use = 'all-MiniLM-L6-v2'
transformer_to_use_with_nouns = 'multi-qa-MiniLM-L6-cos-v1'
datafile = './data/cleaned_data.pkl'
n_gram_range = (1, 2)


topic_list = ['Location','Cleanliness', 'Service', 'Food', 'Value', 
              'Restaurant', 'Room', 'Friendly staff', 'Room service', 
              'Walking distance']

