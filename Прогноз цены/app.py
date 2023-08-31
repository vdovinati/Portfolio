# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:17:10 2023

@author: tti
"""

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import plotly.express as px

st.title('Онлайн оценка стоимости вашего автомобиля')
st.header('Сколько стоит ваш автомобиль сейчас')

test = pd.read_csv('test.csv')
data = pd.read_csv('train.csv')
model = pickle.load('cb.pickle')
years = np.unique(data.year)
brand = np.unique(data.make.astype('str'))
model = np.unique(data.model.astype('str'))
odometer = list(range(1000, 400000, 5000))
condition = np.unique(data.condition.astype('str'))
color = np.unique(data.color.astype('str'))
compl = np.unique(data.interior.astype('str'))

st.sidebar.markdown("## Узнай сколько стоит твое авто")
select_event_2 = st.sidebar.selectbox(
    'Какая марка у вашей машины?', brand.tolist())

select_event3 = st.sidebar.selectbox(
    'Какая модель у вашей машины?', model.tolist())

select_event_4 = st.sidebar.selectbox(
    'Какой год выпуска у вашей машины?', years.tolist())

select_event_6 = st.sidebar.selectbox(
    'Какой пробег у вашей машины?', odometer)


select_event_7 = st.sidebar.selectbox(
    'Какое состояние у вашей машины?', condition)

select_event_8 = st.sidebar.selectbox(
    'Какой цвет у вашей машины?', color)

select_event_9 = st.sidebar.selectbox(
    'Какая комплектация у вашей машины?', compl)

st.header('Dataset statistics: ' + str(select_event_2))
datX = data[data.make == select_event_2]
datX = datX.dropna()
st.write(datX.describe())

fig, ax = plt.subplots()
ax = sns.scatterplot(data = datX, x="odometer", y="sellingprice")
st.header('Scatter Plot Chart Cost and Odometer')
st.pyplot(fig)

fig, ax = plt.subplots()
ax = sns.scatterplot(data = datX, x="year", y="sellingprice")
st.header('Scatter Plot Chart Cost and Year')
st.pyplot(fig)

fig, ax = plt.subplots()
ax = sns.scatterplot(data = datX, x="condition", y="sellingprice")
st.header('Scatter Plot Chart Cost and Condition')
st.pyplot(fig)

fig = px.scatter(
    datX,
    x="year",
    y="sellingprice",
    size="condition",
    color="model",
    hover_name="seller",
    log_x=True,
    size_max=60,
)


tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])

with tab1:
    # Use the Streamlit theme.
    # This is the default. So you can also omit the theme argument.
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab2:
    # Use the native Plotly theme.
    st.plotly_chart(fig, theme=None, use_container_width=True)


with st.sidebar:
    if st.button('Дать прогноз'):
        st.write('Получите и распишитесь')
    else:
        st.write('Нажмите для прогноза')
        
with st.sidebar:
    upload_files = st.file_uploader("test.csv", accept_multiple_files = True)
    for upload_file in upload_files:
        bytes_data = uploaded_file.read()
        st.write("test.csv",uploaded_file.name)
        st.write(bytes_data)
predict = cb.predict(bytes_data)
st.header('Ваш авто будет стоить：', predict)

