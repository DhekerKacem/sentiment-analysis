import streamlit as st
import requests

API_URL = "https://predict-app-api.azurewebsites.net/predict"
REPORT_URL = "https://predict-app-api.azurewebsites.net/bad-predict"

st.title('Sentiment Analysis with BERT')

user_input = st.text_area("Enter text for sentiment analysis:", "Type here your tweet")
if st.button('Predict Sentiment'):
    response = requests.post(API_URL, json={'tweet': user_input})
    if response.status_code == 200:
        result = response.json()
        st.write(f'Sentiment: {result["sentiment"]}')
    else:
        st.error("Failed to get prediction from the API.")

if st.button('Report Wrong Prediction'):
    report_response = requests.get(REPORT_URL)
    if report_response.status_code == 406:
        st.success("Error reported successfully.")
    else:
        st.error("Failed to report error.")
