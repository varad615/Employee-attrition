import streamlit as st
from PIL import Image
st.set_page_config(
    page_title="Welcome",
    page_icon="",
)

st.write("# Employee Attrition Prediction using ANN ")



image = Image.open('Employee-Attrition.png')

st.image(image)

st.markdown(
    """
    Welcome to our Employee Attrition Predictor! We're excited to provide you with a tool that can help 
    you better understand and manage employee turnover in your organization!

    
    
    According to the Cambridge dictionary, "Attrition is the process of gradually 
    making something weaker and destroying it." Employee attrition follows the same definition.

Well, almost!

Employee attrition is the reduction of staff when employees leave the organization
 without emphasizing their replacement.


"""
)
st.markdown(" :blue[An employee's motivation is a direct result of the sum of interactions with his or her manager. -Bob Nelson]")

st.markdown("We're confident that our employee attrition prediction calculator will be an invaluable resource for you and your organization, helping you to create a more stable and productive workforce. Thank you for choosing our tool, and we look forward to helping you achieve your HR goals!")