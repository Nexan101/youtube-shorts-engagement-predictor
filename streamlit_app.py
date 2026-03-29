import streamlit as st
import joblib, numpy as np, pandas as pd
import plotly.express as px

model = joblib.load('generated/model.pkl')

st.title('YouTube Shorts Like Ratio Predictor')
st.subheader('Enter the features of the video to predict the like ratio')

duration = st.slider('Video Duration (secs)', 10, 60, 28)
day = st.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
hour = st.slider('Hour Posted (24h)', 0, 23, 18)
days_live = st.number_input('Days Since Posted', min_value=1, max_value=3650, value=30)
title_len = st.slider('Title Length (chars)', 10, 100, 45)

day_num = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(day)

if st.button('Predict Like Ratio'):
    x = [[duration, day_num, hour, days_live, title_len]]
    pred = model.predict(x)[0]
    st.metric('Predicted Like Ratio', f'{pred*100:.1f}%')
    