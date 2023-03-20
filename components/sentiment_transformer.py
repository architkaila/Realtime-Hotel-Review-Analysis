import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from scripts.transformers_infer import run_inference

## Initiate logging
logger = logging.getLogger(__name__)

def sentiment_transformer_UI(unseen_df):
    """
    The main UI function to display the Landing page UI
    """
    st.write("Hello from Sentiment Transformer")

    st.write('''  
        ### Sentiment Analysis using BERT Transformer      
        ''')
    
    hotel_list = [f"Hotel {i}" for i in unseen_df.index]
    
    with st.container():
        selected_hotel_review = st.selectbox("Select a Review", hotel_list)
        selected_hotel_review_index = int(selected_hotel_review.split(" ")[1])
    
    st.write("Original REVIEW")
    st.write(unseen_df.loc[selected_hotel_review_index, 'Review'])
    
    if st.button('Analyse Review'):

        # Fetch the prediction
        pred_class = run_inference(instance=selected_hotel_review_index)
        
        ## Display the sentiment
        st.write("Predicted Sentiment")
        if pred_class[0] == 1:
            st.success("Positive Sentiment", icon="✅")
        else:
            st.error("Negative Sentiment", icon="🚨")