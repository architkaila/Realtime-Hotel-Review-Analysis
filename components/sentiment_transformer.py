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
    
    with st.container():
        selected_hotel_review = st.selectbox("Select a Review", unseen_df.index)
    
    st.write("Original REVIEW")
    st.write(unseen_df.loc[selected_hotel_review, 'Review'])
    
    if st.button('Analyse Review'):

        # Fetch the prediction
        pred_class = run_inference(instance=int(selected_hotel_review))
        
        ## Display the sentiment
        st.write("Predicted Sentiment")
        if pred_class[0] == 1:
            st.success("Positive Sentiment", icon="✅")
        else:
            st.error("Negative Sentiment", icon="🚨")