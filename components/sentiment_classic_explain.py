## Library imports
import streamlit as st
from scripts.random_forest_infer import run_pipiline
import matplotlib.pyplot as plt


def sentiment_classic_explain_UI(unseen_df):
    """
    The main UI function to display the page UI for animal classification
    """
    st.write('''  
        ### Sentiment Analysis using TF-IDF and Random Forest with Explainability      
        ''')
    
    with st.container():
        selected_hotel_review = st.selectbox("Select a neighbourhood", unseen_df.index)
    

    st.write("RAW REVIEW")
    st.write(unseen_df.loc[selected_hotel_review, 'Review'])
    
    if st.button('Analyse Review'):
        force_plot, shap_values, top_keywords, top_adjective, review_corrected_text, top_adjective_shap, remaining_adjectives = run_pipiline(instance=int(selected_hotel_review))

        st.pyplot(fig=force_plot ,dpi=200, pad_inches=0)
        plt.clf()