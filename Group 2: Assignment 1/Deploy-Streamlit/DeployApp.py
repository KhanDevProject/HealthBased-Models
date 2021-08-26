 

import pickle
import streamlit as st
import numpy as np

 
# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache(allow_output_mutation=True)

# defining the function which will make the prediction using the data which the user inputs 
def prediction(Age, Gender,EDUC,SES,MMSE,CDR,eTIV,nWBV,ASF):
    
    # preprocess user input
    if Gender == 'Male':
        Gender = 0
    else:
        Gender = 1
        
    # making predictions of all the grouphs in the file
    predictions = classifier.predict(
        [[Age,Gender,EDUC,SES,MMSE,CDR,eTIV,nWBV,ASF]])
    
    class_names = ['Converted','Demented','Nondemented']
    final_pred = class_names[np.argmax(predictions)]        
         
    return final_pred, predictions
# main function defines our webpage
def main():
     # front end elements of the web page 
    html_temp ="""
    <div style ="background-color:red;background-image: linear-gradient(45deg, #f3ec78, #af4261);background-size: 100%;  background-repeat: repeat;;
    -webkit-background-clip: text;-webkit-text-fill-color: transparent; -moz-background-clip: text;
    -moz-text-fill-color: transparent;"> 
    <h1 style ="text-align: center;font-family: "Archivo Black", sans-serif;font-weight: normal;font-size: 6em; ">Alzheimer Deployment</h1> 
    </div> """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    
    # allow user input 
    Gender = st.selectbox('Gender',('Male','Female'))
    Age  = st.slider('Age',min_value=1, max_value=100, value=10, step=1)
    EDUC  = st.slider('Years of Education: EDUC',min_value=1, max_value=100, value=1, step=1)
    SES  = st.slider('Socialeconomic Status (SES): 1-5',min_value=1, max_value=50, value=1, step=1)
    MMSE = st.slider('Mini Mental State Examination (MMSE)',min_value=1, max_value=40, value=1, step=1)
    CDR = st.slider('Clinical Dimentia Rating (CDR): 0-3',min_value=0.0, max_value=3.0, value=0.0, step=0.5)
    eTIV = st.slider(' Estimated total intracranial volume:eTIV ',min_value=1000, max_value=2000, value=1000, step=1)
    nWBV  = st.slider('Normalized whole Brain Volume: nWBV',min_value=0.0, max_value=1.0, value=0.0,step=0.01)
    ASF  = st.slider('Atlas Scaling Factor: ASF',min_value=0.7, max_value=1.6, value=0.7, step=0.01)
    # Make the prediction and store it when clicked
    if st.button("Predict"):
        result, preds = prediction(Age,Gender,EDUC,SES,MMSE,CDR,eTIV,nWBV,ASF)
        st.success(f'Health Status is {result}')
        
if __name__=='__main__': 
    main()
