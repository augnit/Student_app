import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder,StandardScaler



def load_model():
    with open("Studenet_final_lr_model.pkl",'rb') as file:
        linear,scaler,le = pickle.load(file)
    return linear,scaler,le


# funtion to take a data and preprocessing it 
def pre_input_data(placement, scaler ,le):
    placement['Extracurricular Activities'] = le.transform([placement['Extracurricular Activities']])[0]
    df = pd.DataFrame([placement])
    df_transform = scaler.transform(df)
    return df_transform


# predict data
def predict_data(placement):
    linear,scaler,le = load_model()
    pre_data = pre_input_data(placement, scaler ,le)
    prediction = linear.predict(pre_data)
    return prediction

#streamlit UI
def main():
    st.title("Student performance prediction")
    st.write("Enter your data")

    Hour_Study = st.number_input("Hour Study",min_value = 1,max_value = 10,value = 5)
    Previous_Score = st.number_input("Previous Score",min_value = 40,max_value = 100,value = 50)
    Extra_caricular_activity=st.selectbox("Extra caricular activity",['Yes','No'])
    Sleeping_hours= st.number_input("Sleeping hours",min_value = 4,max_value = 10,value = 5)
    Number_of_ques_solved = st.number_input("Number of question paper solved",min_value = 0,max_value = 10,value = 5)

    if st.button("Predict data"):
        user_data = {
            'Hours Studied' : Hour_Study ,
            'Previous Scores' : Previous_Score ,
            'Extracurricular Activities' : Extra_caricular_activity,
            'Sleep Hours' : Sleeping_hours,	
            'Sample Question Papers Practiced': Number_of_ques_solved   
        }
        
        st.write(user_data)
        predict = predict_data(user_data)
        st.success(f"Your prediction result is {predict}")
    
    

if __name__ == "__main__":
    main()

    

