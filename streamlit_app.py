
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
# Title
st.title('Liver Disease Prediction')

# Image
#img=Image.open('liver2.PNG')
#st.image(img, width=500,channels='RGB')

# Load Cleaned Data
df = pd.read_csv('cleaned_data.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)
#st.dataframe(df.head())
# Load Preprocessor

sc= pickle.load(open('sc.pkl', 'rb'))
pca= pickle.load(open('pca.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

Age=st.text_input('Age')
Gender=st.selectbox('Gender',['Male','Female'])
Total_Bilirubin=st.text_input('Total Bilirubin')
Direct_Bilirubin=st.text_input('Direct Bilirubin')
Alkaline_Phosphotase=st.text_input('Alkaline Phosphotase')
Alamine_Aminotransferase=st.text_input('Alamine Aminotransferase')
Aspartate_Aminotransferase=st.text_input('Aspartate Aminotransferase')
Total_Protiens=st.text_input('Total Protiens')
Albumin=st.text_input('Albumin')
Albumin_and_Globulin_Ratio=st.text_input('Albumin and Globulin Ratio')
#numerical_cols = []

#X_dic = {}
#X_new = pd.DataFrame()
#X_new['Owner_Type'] = ordinal_encoder.transform(X_new[['Owner_Type']])
#X_new = encoder.transform(X_new)
#X_new[numerical_cols] = scaler.transform(X_new[numerical_cols])
#numerical_cols = []
new_data={'Age': Age ,'Gender': Gender,'Total_Bilirubin':Total_Bilirubin, 'Direct_Bilirubin':Direct_Bilirubin,'Alkaline_Phosphotase':Alkaline_Phosphotase,'Alamine_Aminotransferase':Alamine_Aminotransferase,'Aspartate_Aminotransferase':Aspartate_Aminotransferase,'Total_Protiens':Total_Protiens,'Albumin':Albumin,'Albumin_and_Globulin_Ratio':Albumin_and_Globulin_Ratio}
new_data = pd.DataFrame(new_data, index=[0])

new_data['Age'] = pd.to_numeric(new_data['Age'], errors='coerce')
new_data['Total_Bilirubin'] = pd.to_numeric(new_data['Total_Bilirubin'], errors='coerce')
new_data['Direct_Bilirubin'] = pd.to_numeric(new_data['Direct_Bilirubin'], errors='coerce')
new_data['Alkaline_Phosphotase'] = pd.to_numeric(new_data['Alkaline_Phosphotase'], errors='coerce')
new_data['Alamine_Aminotransferase'] = pd.to_numeric(new_data['Alamine_Aminotransferase'], errors='coerce')
new_data['Aspartate_Aminotransferase'] = pd.to_numeric(new_data['Aspartate_Aminotransferase'], errors='coerce')
new_data['Total_Protiens'] = pd.to_numeric(new_data['Total_Protiens'], errors='coerce')
new_data['Albumin'] = pd.to_numeric(new_data['Albumin'], errors='coerce')
new_data['Albumin_and_Globulin_Ratio'] = pd.to_numeric(new_data['Albumin_and_Globulin_Ratio'], errors='coerce')
new_data = new_data.dropna()



if new_data.empty:
    st.write("Please provide valid input data.")
else:
    # Map Gender values
    new_data = pd.DataFrame(new_data, index=[0])
    new_data.Gender = new_data.Gender.map({'Male': 0, 'Female': 1})

    # Preprocess the input data
    new_data_preprocessed = sc.transform(new_data)
    #new_data_preprocessed =pca.transform(new_data)
    #new_data_preprocessed = sc.named_transformers_['num'].transform(new_data)
    # Make the prediction
    prediction = model.predict(new_data_preprocessed)

    # Display the prediction
    if prediction[0] == 1:
        st.write("Based on the entered information, it is predicted that the person has liver disease.")
    else:
        st.write("Based on the entered information, it is predicted that the person does not have liver disease.")
