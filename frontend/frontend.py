import streamlit as st
import requests

# Page Configuration
st.set_page_config(
    page_title="Cyber Threat Detector",
    page_icon="🛡️",
    layout="centered",
)

#API Configuration
API_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000/predict/")

#UI Design 
st.title("🛡️ Text-Based Cyber Threat Detector")
st.markdown(
    "Enter a piece of text (e.g., from an email, report, or log) to classify it as **Malicious** or **Benign**."
)

#Input Area
user_input = st.text_area(
    "Enter text for analysis:",
    height=150,
    placeholder="e.g., The malware downloads additional payloads and steals user credentials."
)

#Prediction Button and Logic 
if st.button("Analyze Text", type="primary"):
    if user_input:
        with st.spinner("Analyzing..."):
            try:
                # Prepare the payload for the API
                payload = {"text": user_input}
                
                # Make the request to the FastAPI backend
                response = requests.post(API_URL, json=payload)
                # Raise an exception for bad status codes
                response.raise_for_status()   
                
                result = response.json()
                
                # Display the result
                st.subheader("Analysis Result")
                label = result.get("prediction")
                confidence = result.get("confidence", 0)
                
                if label == "Malicious":
                    st.error(f"**Prediction:** {label}")
                    st.progress(confidence)
                    st.metric(label="Confidence Score", value=f"{confidence:.2%}")
                else:
                    st.success(f"**Prediction:** {label}")
                    # Show confidence for benign
                    st.progress(1 - confidence)  
                    st.metric(label="Confidence Score", value=f"{1 - confidence:.2%}")
                
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the API. Please ensure the backend is running. Details: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter some text to analyze.")

 
st.markdown("---")
st.markdown("Built with FastAPI & Streamlit")