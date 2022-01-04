import streamlit as st

st.title("Housing Prices Prediction")

st.write(""" ### Project description 

We have trained several models to predict the price of a house based on features such as the area of the house and the condition and quality of their different rooms. """)

import pickle
model = pickle.load(open('models/trained_pipe_knn.sav', 'rb'))

new_house = pd.DataFrame({
    'LotArea':[9000],
    'TotalBsmtSF':[1000], 
    'BedroomAbvGr':[5], 
    'GarageCars':[4]
})

prediction = loaded_model.predict(new_house)

st.write("The price of the house is ", prediction)