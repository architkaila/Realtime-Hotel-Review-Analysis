#importing libaries
import re, numpy as np, pandas as pd
from pprint import pprint

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import pickle

from matplotlib import pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from collections import Counter

from config import stop_words
from config import num_lda_topics, passes, chunksize, iterations

def sent_to_words(sentences):
    """
    Convert a collection of sentences to a list of words

    Args:
        sentences (list): list of sentences
    
    Returns:
        list: list of words
    """

    for sent in sentences:
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  


# !python3 -m spacy download en  # run in terminal once
def process_words(texts, bigram_mod, trigram_mod, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """
    Remove Stopwords, Form newly created Bigrams, Trigrams and Lemmatization.

    Args:
        texts (list): list of words from sentences
        bigram_mod (gensim.models.phrases.Phraser): bigram model
        trigram_mod (gensim.models.phrases.Phraser): trigram model
        stop_words (list, optional): list of stopwords. Defaults to stop_words.
        allowed_postags (list, optional): list of allowed postags. Defaults to ['NOUN', 'ADJ', 'VERB', 'ADV'].
    
    Returns:
        list: updated list of words
    """
    # Get bigrams and trigrams
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    texts_out = []

    # Initialize spacy 'en' model
    nlp = spacy.load("en_core_web_sm")

    # Do lemmatization keeping only noun, adj, vb, adv
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    
    # Remove stopwords once more after lemmatization of BIgrams and Trigrams
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    
    return texts_out

def format_topics_sentences(ldamodel, corpus, texts):
    """
    Format topics and sentences

    Args:
        ldamodel (gensim.models.ldamodel.LdaModel): LDA model
        corpus (list): list of corpus
        texts (list): list of words from sentences
    
    Returns:
        pd.DataFrame: dataframe of topics and sentences
    """

    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return(sent_topics_df)

def build_n_gram_models(data_words):
    """
    Build the bigram and trigram models

    Args:
        data_words (list): list of words from sentences

    Returns:
        bigram_mod, trigram_mod: bigram and trigram models for gensim
    """
    # Build the bigram and trigrams
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # Set higher threshold fewer phrases
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    
    # Build bigram and trigram models
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    return bigram_mod, trigram_mod

def word_cloud(lda_model):
    """
    Wordcloud of Top N words in each topic

    Args:
        lda_model (gensim model): trained LDA model
    
    Returns:
        None
    """

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    cloud = WordCloud(stopwords=stop_words,
                    background_color='white',
                    width=2500,
                    height=1800,
                    max_words=10,
                    colormap='tab10',
                    color_func=lambda *args, **kwargs: cols[i],
                    prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig('./results/lda_topics_wordcloud.png')
    #plt.show()

def plot_word_count(lda_model):
    """
    Plot Word Count and Weights of Topic Keywords

    Args:
        lda_model (gensim model): trained LDA model
    
    Returns:
        None
    """

    # Get topic keywords
    topics = lda_model.show_topics(formatted=False)

    # Flatten list of words
    data_flat = [w for w_list in data_ready for w in w_list]
    
    # Count word frequencies
    counter = Counter(data_flat)

    out = []
    # For each topic, get the keywords and their weights
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])

    # Convert to dataframe
    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, dpi=160)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
        ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)    
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
    plt.savefig('./results/lda_word_counts_of_each_topic.png')
    #plt.show()

def run_pipeline():
    """
    Run the pipeline to create the LDA model and generate the wordcloud and word count plots
    for the topics

    Args:
        None
    
    Returns:
        None
    """
    # Load Data
    print("[INFO] Loading Data...")
    data = pd.read_pickle("./data/clean_lemmatized_data.pkl")

    # Convert to list
    data = data.cleaned_Review.values.tolist()

    # Get list of words from sentences
    data_words = list(sent_to_words(data))
    
    # Build the bigram and trigram models
    print("[INFO] Building the bigram and trigram models...")
    bigram_mod, trigram_mod = build_n_gram_models(data_words)

    # Remove Stop Words, Form Bigrams, Trigrams and Lemmatization
    print("[INFO] Removing Stop Words, Forming Bigrams, Trigrams and Lemmatizing...")
    data_ready = process_words(data_words, bigram_mod, trigram_mod)

    # Create Dictionary
    print("[INFO] Creating Corpus Dictionary...")
    id2word = corpora.Dictionary(data_ready)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]

    # Build LDA model
    print("[INFO] Building LDA model...")
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_lda_topics, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=chunksize,
                                            passes=passes,
                                            alpha='symmetric',
                                            iterations=iterations,
                                            per_word_topics=True)

    # Print the identified Keyword in the topics defined by LDA
    pprint(lda_model.print_topics())
    
    # Save the model
    print("[INFO] Saving the LDA model...")
    pickle.dump(lda_model, open('./models/lda_model.sav', 'wb'))

    # Format Identified Topics and Sentences
    print("[INFO] Formatting Identified Topics and Sentences...")
    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Group top sentences under each topic
    sent_topics_sorteddf_mallet = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                                grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                                axis=0)

    # Reset Index    
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    # Format columns
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

    # Show identified topics and keywords
    print(sent_topics_sorteddf_mallet.head(10))

    # Save the identified topics and keywords
    print("[INFO] Saving the identified topics and keywords...")
    sent_topics_sorteddf_mallet.to_csv("./results/lda_topics.csv", index=False)

    # Generate Word Clouds
    print("[INFO] Generating Word Clouds...")
    word_cloud(lda_model)

    # Plot Word Count and Weights of Topic Keywords
    print("[INFO] Plotting Word Count and Weights of Topic Keywords...")
    plot_word_count(lda_model)


if __name__ == "__main__":
    run_pipeline()