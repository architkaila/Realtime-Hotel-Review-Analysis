from nltk.corpus import stopwords

## Stop words from nltk
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 
                   'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see',
                   'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 
                   'right', 'line', 'even', 'also', 'may', 'take', 'come'])

## Unigram mode for tfidf
n_gram_range_tfidf = (1, 1)

## Class weights for Random forest
class_weights = {0:1, 1:6}


## Which sample from test data we want to visualize and explain
explain_sample = 0