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

def reviews_UI(df):
    """
    The main UI function to display the Reviews EDA page UI
    """

    st.markdown("""---""")

    matplotlib.use("agg")
    _lock = RendererAgg.lock
    sns.set_style("darkgrid")

    data_to_plot = {
        "Total Reviews":len(df),
        "Positive Reviews":len(df[df["Sentiment"] == 1]),
        "Negative Reviews":len(df[df["Sentiment"] == 0]),
        "Avg Words / Review":100,
    }



    ## Add view cards for basic information around data
    col1, col2, col3, col4 = st.columns(4)
    columns = [col1, col2, col3, col4]
    
    count = 0
    for key, value in data_to_plot.items():
        with columns[count]:
            st.metric(label= key, value = value)
            count += 1
            if count >= 4:
                count = 0

    row_1_col1, row_1_col2 =  st.columns(2)
    with row_1_col1, _lock:
        st.subheader("Sentiment Distribution")

        ## Pie chart to explore distribution of shows and no shows
        fig = Figure()
        ax = fig.subplots()
        df["Sentiment"].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)
    
    with row_1_col2, _lock:
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
    
    row_2_col1, row_2_col2 =  st.columns(2)
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