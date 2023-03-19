## Library imports
import pandas as pd 
import numpy as np 
import streamlit as st
from streamlit import runtime

## Local imports
from components import reviews_eda
from components import sentiment_transformer
from components import sentiment_classic_explain
from components import about_us
from config import PAGES

@st.cache
def load_data():
    """ 
    Loads the required dataframe into the webapp 

    Args:
        None

    Returns:
        df (pd.DataFrame): The dataframe containing the cleaned data
        unseen_df (pd.DataFrame): The dataframe containing the unseen data
    """

    df = pd.read_pickle("./data/cleaned_data.pkl")
    unseen_df = pd.read_pickle("./data/X_test.pkl")

    return df, unseen_df

## Set the page tab title
st.set_page_config(page_title="Hotel Review Analysis", page_icon="🤖")

## create dataframe from the load function 
df, unseen_df = load_data()

## Landing page UI
def run_UI():
    """
    The main UI function to display the Landing page UI
    """

    ## Set the page title and navigation bar
    st.sidebar.title('Select Menu')
    if st.session_state["page"]:
        page=st.sidebar.radio('Navigation', PAGES, index=st.session_state["page"])
    else:
        page=st.sidebar.radio('Navigation', PAGES, index=0)
    st.experimental_set_query_params(page=page)


    ## Display the page selected on the navigation bar
    if page == 'About Us':
        st.sidebar.write("""
            ## About
            
            About Us
        """)
        st.title("About Us")
        about_us.about_us_UI()

    elif page == 'Reviews EDA':
        st.sidebar.write("""
            ## About
            
            The goal of this project is to perform sentiment analysis on the reviews of Hotels in the US. There are many existing approaches to this. Our novel way brings in explananbility layer over the sentiment classification. We use the SHAP library to explain the model.
        """)
        reviews_eda.reviews_UI(df)

    elif page == 'Transformer Sentiment Analysis':
        st.sidebar.write("""
            ## About
            
            The goal of this project is to perform sentiment analysis on the reviews of Hotels in the US. There are many existing approaches to this. Our novel way brings in explananbility layer over the sentiment classification. We use the SHAP library to explain the model.
        """)
        sentiment_transformer.sentiment_transformer_UI(unseen_df)

    else:
        st.sidebar.write("""
            ## About
            
            The goal of this project is to perform sentiment analysis on the reviews of Hotels in the US. There are many existing approaches to this. Our novel way brings in explananbility layer over the sentiment classification. We use the SHAP library to explain the model.
        """)
        sentiment_classic_explain.sentiment_classic_explain_UI(unseen_df)


if __name__ == '__main__':
    ## Load the streamlit app with "Animal Classifier" as default page
    if runtime.exists():

        ## Get the page name from the URL
        url_params = st.experimental_get_query_params()

        if 'loaded' not in st.session_state:
            if len(url_params.keys()) == 0:
                ## Set the default page as "Animal Classifier"
                st.experimental_set_query_params(page='Reviews EDA')
                url_params = st.experimental_get_query_params()
                
            ## Set the page index
            st.session_state.page = PAGES.index(url_params['page'][0])
        
        ## Call the main UI function
        run_UI()