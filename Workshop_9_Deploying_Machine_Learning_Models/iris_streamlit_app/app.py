# Streamlit is an open-source Python library that transforms data scripts into interactive web applications,
# ideal for deploying machine learning models. Its simplicity allows for rapid prototyping, integrating
# real-time input handling, visualizations, and pre-built UI elements (like sliders and buttons) without 
# requiring web development skills. With easy integration of popular Python libraries and flexible hosting options,
# Streamlit enables accessible and collaborative sharing of ML models with stakeholders.

# Streamlit and FastAPI serve different purposes, though both are excellent tools for deploying ML models.
# Streamlit is designed for rapid prototyping with minimal code, focusing on building interactive, user-friendly 
# web applications with built-in UI components (like sliders, buttons, and graphs) ideal for data science and ML demos.
# It requires no HTML or CSS, making it accessible for data scientists without a web development background.

# FastAPI, on the other hand, is a high-performance framework for building robust APIs with asynchronous support.
# It is better suited for production-grade applications where backend API endpoints are needed, such as serving
# complex ML models for high-load environments. FastAPI is more flexible for custom endpoints and integrations
# with other web technologies but requires additional tools (like frontend frameworks) to build a complete web UI.

# In short, Streamlit is typically better for quick, interactive ML model demos and user interfaces, while FastAPI 
# excels in building scalable, production-ready API services with high-performance requirements.


import streamlit as st
import joblib
import numpy as np
import pandas as pd

# The `st.set_page_config()` function in Streamlit is used to set up the initial configuration for the web app.
# Here, the page title is set to "Iris Flower Classifier", which will appear in the browser tab.
# The `page_icon` parameter assigns a flower emoji 🌸 as the favicon in the browser tab.
# `layout="centered"` centers the content on the page, making it visually balanced, especially for small datasets or simple interfaces.
# `initial_sidebar_state="expanded"` ensures the sidebar is visible by default, providing easy access to options or inputs for users.
# Overall, this configuration enhances the user experience with a clean and organized layout.
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded",
)

# The `@st.cache_resource` decorator is used to cache the output of the `load_model()` function.
# This means that the model will only be loaded once, reducing loading time on subsequent calls and improving performance.
# Caching is particularly useful for large or complex models, as it avoids reloading the model every time the app reruns.
# Here, `load_model()` loads the pre-trained Iris classifier from the 'iris_classifier.pkl' file using `joblib`.
# When `model = load_model()` is called, it retrieves the cached model, ensuring efficient and fast access for predictions.
@st.cache_resource  
def load_model():
    return joblib.load('iris_classifier.pkl')

model = load_model()

# Add a title and description
st.title("🌸 Iris Flower Classification")
st.write("""
### Welcome to the Iris Flower Classifier!
This application predicts the species of Iris flowers based on their measurements.
""")

# The `st.sidebar.header()` function adds a header to the sidebar in the Streamlit app.
# Here, "Input Measurements" is displayed as a header in the sidebar, guiding users to input the necessary measurements 
# for the Iris Flower Classifier model. The sidebar layout is ideal for organizing input fields separately from main content,
# making the app more intuitive and visually organized.
st.sidebar.header('Input Measurements')

# The `user_input_features()` function captures input values for each feature required by the Iris classifier model.
# Each feature (sepal length, sepal width, petal length, and petal width) is collected using `st.sidebar.slider()`,
# which allows users to select values within specified ranges (e.g., 4.0 to 8.0 for sepal length).
# The default values (e.g., 5.4 for sepal length) serve as initial inputs when the app loads.

# The input values are stored in a dictionary `data`, which is then converted into a Pandas DataFrame `features`.
# This DataFrame structure ensures compatibility with the model's expected input format.
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.0, 8.0, 5.4)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.5, 3.4)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 7.0, 1.3)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

# `user_input_features()` returns the DataFrame, and `df = user_input_features()` assigns the input features to `df`,
# which can be used to make predictions with the loaded model.
df = user_input_features()

st.subheader('User Input Parameters') # This provides a clear, labeled section where users can review the values they’ve input.
st.write(df) # Showing this DataFrame helps users verify their inputs before running the model, enhancing the app's interactivity and transparency.

# Make prediction
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

# Map predictions to species names
species_names = ['Setosa', 'Versicolor', 'Virginica']
predicted_species = species_names[prediction[0]]

# Show predictions
st.subheader('Prediction')
st.write(f'**Species**: {predicted_species}')

# Show prediction probabilities
st.subheader('Prediction Probability')
prob_df = pd.DataFrame(prediction_proba, columns=species_names)
st.write(prob_df)

# Add a visualization of the probabilities
st.bar_chart(prob_df.T) # By transposing the DataFrame with `.T`, the chart displays each class as a separate bar, making it easy to interpret the model's confidence levels.


# Add some information about the species
st.subheader('Iris Species Information')
species_info = {
    'Setosa': 'Iris setosa is known for its distinct appearance with small petals.',
    'Versicolor': 'Iris versicolor, also known as the blue flag, has medium-sized petals.',
    'Virginica': 'Iris virginica, also known as the Virginia iris, has large petals.'
}

for species, info in species_info.items():
    with st.expander(f"About {species}"):
        st.write(info)

# Add footer
st.markdown("""
---
Created with ❤️ using Streamlit
""")