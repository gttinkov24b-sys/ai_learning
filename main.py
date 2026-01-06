import streamlit as st
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION & STATE ---
st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")
st.title("✍️ Handwritten Digit Recognition")

# Initialize session state variables
if 'ai_score' not in st.session_state:
    st.session_state.ai_score = 0
if 'prediction_active' not in st.session_state:
    st.session_state.prediction_active = False

# --- 2. MODEL LOADING & INITIAL TRAINING ---
@st.cache_resource
def load_initial_model():
    try:
        # Load sklearn's built-in digits dataset
        digits = load_digits()
        X = digits.images.reshape((len(digits.images), -1)) / 16.0
        y = digits.target
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the MLPClassifier
        model = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=100,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

if 'model' not in st.session_state:
    st.session_state.model = load_initial_model()

model = st.session_state.model

# Sidebar for Score and Instructions
st.sidebar.metric("AI Reputation Score", st.session_state.ai_score)
st.sidebar.header("Instructions")
st.sidebar.write("1. Upload a digit.\n2. Reward or Punish the AI.\n3. The app loops for the next digit.")

# --- 3. THE LOOP LOGIC ---
# If a prediction is NOT active, show the uploader
if not st.session_state.prediction_active:
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        st.session_state.current_image = Image.open(uploaded_file)
        st.session_state.prediction_active = True
        st.rerun()

# If a prediction IS active, show results and feedback, then clear
else:
    image = st.session_state.current_image
    st.image(image, caption='Current Digit', width=200)
    
    try:
        # Preprocessing
        img_gray = image.convert('L')
        img_resized = img_gray.resize((8, 8))
        img_array = np.array(img_resized)
        if np.mean(img_array) > 128:
            img_array = 255 - img_array
        img_array = img_array / 16.0
        img_flat = img_array.flatten().reshape(1, -1)

        # Prediction
        prediction = model.predict(img_flat)[0]
        st.write(f"## AI Prediction: **{prediction}**")

        st.divider()
        st.subheader("Feedback Loop")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Correct (Reward +5)"):
                model.partial_fit(img_flat, [prediction])
                st.session_state.ai_score += 5
                # Reset state to loop back to uploader
                st.session_state.prediction_active = False
                st.rerun()

        with col2:
            actual_digit = st.number_input("Real digit?", 0, 9, step=1)
            if st.button("❌ Wrong (Punish -5)"):
                model.partial_fit(img_flat, [actual_digit])
                st.session_state.ai_score -= 5
                # Reset state to loop back to uploader
                st.session_state.prediction_active = False
                st.rerun()
                
    except Exception as e:
        st.error(f"Error: {e}")
        if st.button("Reset App"):
            st.session_state.prediction_active = False
            st.rerun()
