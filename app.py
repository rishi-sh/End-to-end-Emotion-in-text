#core pkg 
from os import pipe
from sys import path_importer_cache
from typing import Collection
import streamlit as st
import altair as alt

#EDA pkg
import numpy as np
import pandas as pd

import joblib

#loading pipeline
pipe_LR = joblib.load(open("emotion_classifier_pipe_LR_01_january_2022.pkl","rb"))

#function to read the emotion
def predict_emotions(docx):
    results = pipe_LR.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_LR.predict_proba([docx])
    return results


def main():
    st.title("Emotion-Classifier ")
    menu = ["Home"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Emotion In Text")
        with st.form(key='Emotion_clf_form'):
            raw_text = st.text_area("type here")
            submit_text=st.form_submit_button(label='Submit')
        if submit_text:
            Col1,Col2 = st.columns(2)

            #applying the function
            prediction = predict_emotions(raw_text)
            probability= get_prediction_proba(raw_text)
            with Col1:
                st.success("Original Text ")
                st.write(raw_text)
                st.success("Prediction")
                st.write(prediction)
                st.write("Confidence:{}".format(np.max(probability)))
             
            with Col2:
                st.success("Preiction Probability")
                #st.write(probability)
                proba_df = pd.DataFrame(probability,columns=pipe_LR.classes_)
                st.write(proba_df.T)
                #plotting the graph
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions","probability"]
                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
                st.altair_chart(fig,use_container_width=True)

       
    
    elif choice == "Monitor":
        st.subheader("Monitor App")

    else:
        st.subheader("About")





if __name__ == '__main__':
    main()