#!/usr/bin/env python
# coding: utf-8

import streamlit as st

st.title("Housing Prices Prediction")

st.write("""
### Project description
We have trained several models to predict the price of a house based on features such as the area of the house and the condition and quality of their different rooms.

""")

LotArea = st.sidebar.number_input("Lot Area")
TotalBsmtSF = st.sidebar.number_input("Basement Square Feet")
BedroomAbvGr = st.sidebar.number_input("Number of Bedrooms")
GarageCars = st.sidebar.number_input("Car spaces in Garage")

import pandas as pd

new_house = pd.DataFrame({
    'LotArea':[LotArea],
    'TotalBsmtSF':[TotalBsmtSF], 
    'BedroomAbvGr':[BedroomAbvGr], 
    'GarageCars':[GarageCars]
})

import pickle
model = pickle.load(open('models/trained_pipe_knn.sav', 'rb'))

def price_pred(data):
    if model.predict(data) == 0:
        return "That looks like a cheap house..."
    else:
        return "Jesus that house must be expensive!"

    

st.write("#### Predicted Price: ", price_pred(new_house))



