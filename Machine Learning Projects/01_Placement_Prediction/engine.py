import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Placement Predictor", page_icon="💼", layout="centered")

model = pickle.load(open('Machine Learning Projects/01_Placement_Prediction/artifacts/model.pkl', 'rb'))
scaler = pickle.load(open('Machine Learning Projects/01_Placement_Prediction/artifacts/scaler.pkl', 'rb'))

st.title("💼 Campus Placement Prediction Engine")
st.markdown("Check your placement probability by entering your academic and aptitude data on this dashboard.")
st.markdown("---")

cgpa = st.slider("Enter Your cgpa",min_value=3.0,max_value=10.0,value=7.0,step=0.1)
apt_score = st.slider("Enter your Aptitude Test Score",min_value=0,max_value=100,value=45,step=1)

if st.button("Predict Placement Status", type="primary", use_container_width=True):

         user_data = np.array([[cgpa,apt_score]])
         user_data_scale = scaler.transform(user_data)
         prediction = model.predict(user_data_scale)

         if prediction[0] == 1:
                st.success(f"🎉 **Congratulations!** High placement probability cleared. (CGPA: {cgpa}, Score: {apt_score})")
                st.balloons() 
         else:
                st.error(f"❌ **Result: Not Placed.** Current metrics are below the historical selection boundary.")
