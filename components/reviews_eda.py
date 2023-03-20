## Library Imports
import numpy as np
import streamlit as st
import pandas as pd
import pickle

## Imports for plots
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure

from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from config import stop_words


def generate_word_cloud(lda_model, topic_num):
    """
    Function to generate wordclouds for each topic from LDA Model

    Args:
        lda_model (gensim model): LDA model
        topic_num (int): Topic number to generate wordcloud for
    
    Returns:
        cloud (wordcloud): Wordcloud for the topic
    """

    ## Get colors for the wordcloud
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    ## Generate wordclouds for each topic
    cloud = WordCloud(stopwords=stop_words,
                    background_color='white',
                    width=2500,
                    height=1800,
                    max_words=10,
                    colormap='tab10',
                    color_func=lambda *args, **kwargs: cols[topic_num],
                    prefer_horizontal=1.0)

    ## Get the topics from the LDA model
    topics = lda_model.show_topics(formatted=False)

    ## Get the words and their frequency for the topic
    topic_words = dict(topics[topic_num][1])
    
    ## Generate the wordcloud
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    
    return cloud
   


def reviews_UI(df):
    """
    The main UI function to display the Reviews EDA page UI

    Args:
        df (pandas dataframe): Dataframe containing the reviews data
    
    Returns:
        None
    """

    ## Set the page title
    st.title("Hotel Reviews Exploratory Data Analysis")
    st.markdown("""---""")

    ## Set the style for the plots
    matplotlib.use("agg")
    _lock = RendererAgg.lock
    sns.set_style("darkgrid")

    ## Add view cards for basic information around data
    col1, col2, col3 = st.columns([2, 1, 1])
    
    ## Pie chart to explore distribution of sentiment
    with col1:
        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(3, 3))
        df["Sentiment"].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)
    
    ## Add metrics for basic information around data
    with col2:
        st.subheader("")
        st.write("")
        st.write("")
        st.metric(label="Total Reviews", value = len(df))
        st.metric(label="Positive Reviews", value = len(df[df["Sentiment"] == 1]))

    ## Add metrics for basic information around data  
    with col3:
        st.subheader("")
        st.write("")
        st.write("")
        st.metric(label="Negative Reviews", value = len(df[df["Sentiment"] == 0]))
        st.metric(label="Avg Words / Review", value = 100)
    
    ## Display the most common unigrams and bigrams
    row_2_col1, row_2_col2 =  st.columns(2)
    ## Display the most common unigrams
    with row_2_col1, _lock:
        st.subheader("Most Common Unigrams")

        with open('./data/most_common_unigram.pkl', 'rb') as f:
            mostCommon_unigrams = pickle.load(f)

        ## Get the most common unigrams and their frequency
        words = []
        freq = []
        for word, count in mostCommon_unigrams:
            words.append(word)
            freq.append(count)

        ## Plot the most common unigrams
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(x=freq, y=words, ax=ax)
        ax.bar_label(ax.containers[0])
        st.pyplot(fig)
    
    ## Display the most common bigrams
    with row_2_col2, _lock:
        st.subheader("Most Common Biigrams")

        ## Get the most common bigrams and their frequency
        mostCommon_bigrams = pd.read_pickle("./data/most_common_bigram.pkl")

        ## Plot the most common unigrams
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(x=mostCommon_bigrams['frequency'], y=mostCommon_bigrams['ngram'], ax=ax)
        ax.bar_label(ax.containers[0])
        st.pyplot(fig)

    ## Display the wordclouds for each topic from LDA
    st.markdown("""---""")
    st.subheader("Topic Modelling LDA Results")

    ## Load the LDA model
    lda_model =  pickle.load(open("./models/lda_model.sav", "rb"))

    ## Display the wordclouds for each topic
    row_3_col1, row_3_col2 =  st.columns(2)
    ## Display the wordcloud for topic 0
    with row_3_col1, _lock:
        st.write("Topic 1")

        ## Generate the wordcloud for the topic 0
        w_cloud_1 = generate_word_cloud(lda_model, topic_num=0)

        ## Plot the wordcloud
        fig, ax = plt.subplots()
        plt.gca().imshow(w_cloud_1)
        plt.gca().axis('off')
        st.pyplot(fig)
    
    ## Display the wordcloud for topic 1
    with row_3_col2, _lock:
        st.write("Topic 2")

        ## Generate the wordcloud for the topic 1
        w_cloud_2 = generate_word_cloud(lda_model, topic_num=1)

        ## Plot the wordcloud
        fig, ax = plt.subplots()
        plt.gca().imshow(w_cloud_2)
        plt.gca().axis('off')
        st.pyplot(fig)

    ## Display the wordcloud for topic 2
    row_4_col1, row_4_col2 =  st.columns(2)
    with row_4_col1, _lock:
        st.write("Topic 3")

        ## Generate the wordcloud for the topic 2
        w_cloud_3 = generate_word_cloud(lda_model, topic_num=2)

        ## Plot the wordcloud
        fig, ax = plt.subplots()
        plt.gca().imshow(w_cloud_3)
        plt.gca().axis('off')
        st.pyplot(fig)
    
    ## Display the wordcloud for topic 3
    with row_4_col2, _lock:
        st.write("Topic 4")

        ## Generate the wordcloud for the topic 3
        w_cloud_4 = generate_word_cloud(lda_model, topic_num=3)

        ## Plot the wordcloud
        fig, ax = plt.subplots()
        plt.gca().imshow(w_cloud_4)
        plt.gca().axis('off')
        st.pyplot(fig)

    ## Display the topic modelling results from candidate noun ropics
    st.markdown("""---""")
    st.subheader("Topic Modelling with Transformers using Noun Topics")
    
    ## Show the sample review
    st.write("Sample Review")
    st.write("unique great stay wonderful time hotel monaco location excellent short stroll main downtown shopping area pet friendly room show sign animal hair smell monaco suite sleep area big striped curtain pull closed nice touch feel cosy goldfish name brandi enjoyed n partake free wine coffee tea service lobby think great feature great staff friendly free wireless internet hotel work suite laptop decor lovely eclectic mix patten color palatte animal print bathrobe feel like rock star nice n look like sterile chain hotel hotel personality excellent stay")
    
    ## Show the identified topics
    st.write("Identified Noun Topics:")
    row_5_col1, row_5_col2, row_5_col3 =  st.columns(3)
    
    with row_5_col1, _lock:
        st.info("luxurious")
    with row_5_col2, _lock:
        st.info("hotels")
    with row_5_col3, _lock:
        st.info("villas")

    ## Display the topic modelling results from predefined candidate topics
    st.markdown("""---""")
    st.subheader("Topic Modelling with Transformers using Predefined Candidate Topics")
    
    ## Show the predefined candidate topics
    st.write ("Predefined Candidate Topics: Location, Cleanliness, Service, Food, Value, Restaurant, Room, Friendly staff, Room service, Walking distance")

    ## Show the sample review
    st.write("Sample Review")
    st.write("unique great stay wonderful time hotel monaco location excellent short stroll main downtown shopping area pet friendly room show sign animal hair smell monaco suite sleep area big striped curtain pull closed nice touch feel cosy goldfish name brandi enjoyed n partake free wine coffee tea service lobby think great feature great staff friendly free wireless internet hotel work suite laptop decor lovely eclectic mix patten color palatte animal print bathrobe feel like rock star nice n look like sterile chain hotel hotel personality excellent stay")
    
    ## Show the identified topics
    st.write("Identified Topics:")
    row_6_col1, row_6_col2, row_6_col3 =  st.columns(3)
    
    with row_6_col1, _lock:
        st.warning("Room")
    with row_6_col2, _lock:
        st.warning("Restaurant")
    with row_6_col3, _lock:
        st.warning("Room service")