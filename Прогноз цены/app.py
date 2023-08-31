# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:47:48 2023

@author: mixal
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


dat = pd.read_csv('train.csv')
#model = pickle.load('model.pickle')
years = np.unique(dat.year)
brand = np.unique(dat.make.astype('str'))
model = np.unique(dat.model.astype('str'))
odometer = list(range(1000, 400000, 5000))
condition = np.unique(dat.condition.astype('str'))
color = np.unique(dat.color.astype('str'))
compl = np.unique(dat.interior.astype('str'))


st.sidebar.markdown("## Нам нужны данные для прогноза!")
select_event_2 = st.sidebar.selectbox(
    'Какая марка у вашей машины?', brand.tolist())

select_event = st.sidebar.selectbox(
    'Какой год выпуска у вашей машины?', years.tolist())

select_event_3 = st.sidebar.selectbox(
    'Какой пробег у вашей машины?', odometer)


select_event_4 = st.sidebar.selectbox(
    'Какое состояние у вашей машины?', condition)

select_event_5 = st.sidebar.selectbox(
    'Какой цвет у вашей машины?', color)

select_event_6 = st.sidebar.selectbox(
    'Какая комплектация у вашей машины?', compl)


# select_event_2 = st.sidebar.selectbox(
#     'What brand car are you selling?', brand.tolist())



st.header('Dataset statistics: ' + str(select_event_2))
datX = dat[dat.make == select_event_2]
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