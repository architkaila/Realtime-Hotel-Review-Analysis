import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

## Initiate logging
logger = logging.getLogger(__name__)

def sentiment_transformer__UI():
    """
    The main UI function to display the Landing page UI
    """
    st.write("Hello from Sentiment Transformer")