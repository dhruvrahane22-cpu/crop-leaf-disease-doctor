import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="Crop Leaf-Disease Doctor", layout="wide")

st.title("🌿 Crop Leaf-Disease Doctor & Advisory Chatbot")
st.markdown("Upload a leaf image to diagnose plant diseases and retrieve treatment guidance.")

# Initialize Session State
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "chat_response" not in st.session_state:
    st.session_state["chat_response"] = None

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    diagnose_btn = st.button("Diagnose Leaf")

if diagnose_btn:
    if uploaded_file is None:
        st.warning("Please upload an image first!")
    else:
        with st.spinner("Analyzing leaf image..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
                data = {"user_query": ""}
                
                response = requests.post("http://127.0.0.1:8000/predict", files=files, data=data)
                
                if response.status_code == 200:
                    st.session_state["prediction"] = response.json()
                    st.session_state["file_bytes"] = uploaded_file.getvalue()
                    st.session_state["file_name"] = uploaded_file.name
                    st.session_state["chat_response"] = None  # Reset chatbot response on new image
                else:
                    st.error(f"Server Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")

if st.session_state["prediction"] is not None:
    result = st.session_state["prediction"]
    
    with col2:
        st.success(f"**Diagnosis:** {result['predicted_class']}")
        st.info(f"**Confidence:** {result['confidence'] * 100:.2f}%")
        
        st.markdown("---")
        st.markdown("### 💬 Ask Advisory Chatbot")
        
        # Wrapped in a dedicated form so submitting query always passes text reliably
        with st.form(key="chatbot_form"):
            user_query = st.text_input("Ask a follow-up question (e.g., 'How do I treat this?' or 'How do I prevent this?'):")
            submit_query = st.form_submit_button("Ask Chatbot")
            
        if submit_query and user_query.strip():
            with st.spinner("Fetching advice..."):
                try:
                    files = {"file": (st.session_state["file_name"], st.session_state["file_bytes"], "image/jpeg")}
                    data = {"user_query": user_query.strip()}
                    
                    response = requests.post("http://127.0.0.1:8000/predict", files=files, data=data)
                    
                    if response.status_code == 200:
                        chat_res = response.json()
                        st.session_state["chat_response"] = {
                            "advisory": chat_res["advisory"],
                            "citation": chat_res["citation"]
                        }
                        st.rerun()
                    else:
                        st.error(f"Server Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")

        # Display chatbot output whenever present
        if st.session_state["chat_response"] is not None:
            chat = st.session_state["chat_response"]
            st.markdown(f"**Response:** {chat['advisory']}")
            st.caption(f"**Source:** {chat['citation']}")