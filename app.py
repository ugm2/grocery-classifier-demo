import os
import streamlit as st
from PIL import Image
import requests
import io
import time
from model import ViTForImageClassification

st.set_page_config(
     page_title="Grocery Classifier",
     page_icon="interface/shopping-cart.png",
     initial_sidebar_state="expanded"
)

@st.cache()
def load_model():
    with st.spinner("Loading model"):
        model = ViTForImageClassification('google/vit-base-patch16-224')
        model.load('model/')
    return model
        
model = load_model()
feedback_path = "feedback"

def predict(image):
    print("Predicting...")
    # Load using PIL
    image = Image.open(image)

    prediction, confidence = model.predict(image)

    return {'prediction': prediction[0], 'confidence': round(confidence[0], 3)}, image

def submit_feedback(correct_label, image):
    folder_path = feedback_path + "/" + correct_label + "/"
    os.makedirs(folder_path, exist_ok=True)
    image.save(folder_path + correct_label + "_" + str(int(time.time())) + ".png")
    
def retrain_from_feedback():
    model.retrain_from_path(feedback_path, remove_path=True)

def main():
    labels = set(list(model.label_encoder.classes_))

    st.title("🍇 Grocery Classifier 🥑")
        
    if labels is None:
        st.warning("Received error from server, labels could not be retrieved")
    else:
        st.write("Labels:", labels)

    image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        st.image(image_file)

        st.subheader("Classification")
        
        if st.button("Predict"):
            st.session_state['response_json'], st.session_state['image'] = predict(image_file)

        if 'response_json' in st.session_state and st.session_state['response_json'] is not None:
            # Show the result
            st.markdown(f"**Prediction:** {st.session_state['response_json']['prediction']}")
            st.markdown(f"**Confidence:** {st.session_state['response_json']['confidence']}")
            
            # User feedback
            st.subheader("User Feedback")
            st.markdown("If this prediction was incorrect, please select below the correct label")
            correct_labels = labels.copy()
            correct_labels.remove(st.session_state['response_json']["prediction"])
            correct_label = st.selectbox("Correct label", correct_labels)
            if st.button("Submit"):
                # Save feedback
                try:
                    submit_feedback(correct_label, st.session_state['image'])
                    st.success("Feedback submitted")
                except Exception as e:
                    st.error("Feedback could not be submitted. Error: {}".format(e))
                    
            # Retrain from feedback
            if st.button("Retrain from feedback"):
                try:
                    retrain_from_feedback()
                    st.success("Model retrained")
                except Exception as e:
                    st.warning("Model could not be retrained. Error: {}".format(e))
                    
main()