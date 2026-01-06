import streamlit as st
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION & STATE ---
st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")
st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a digit and 'train' the AI through rewards and punishments.")

# Initialize session state for score and the live model
if 'ai_score' not in st.session_state:
    st.session_state.ai_score = 0

# --- 2. MODEL LOADING & INITIAL TRAINING ---
@st.cache_resource
def load_initial_model():
    try:
        digits = load_digits()
        # Reshape and normalize (0-16 range to 0-1)
        X = digits.images.reshape((len(digits.images), -1)) / 16.0
        y = digits.target
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        model = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=100,
            random_state=42
        )
        # Initial fit to establish the classes (0-9)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

# Load the model into session state if it's not there
if 'model' not in st.session_state:
    st.session_state.model = load_initial_model()

model = st.session_state.model

# --- 3. UI: SCORE & UPLOADER ---
st.sidebar.metric("AI Reputation Score", st.session_state.ai_score)

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=200)

    # --- 4. IMAGE PROCESSING ---
    try:
        # Preprocessing to match 8x8 sklearn format
        img_gray = image.convert('L')
        img_resized = img_gray.resize((8, 8))
        img_array = np.array(img_resized)

        # Invert if background is white (training data is light-on-dark)
        if np.mean(img_array) > 128:
            img_array = 255 - img_array

        # Normalize and flatten
        img_array = img_array / 16.0
        img_flat = img_array.flatten().reshape(1, -1)

        if model is not None:
            # Make prediction
            prediction = model.predict(img_flat)[0]
            probs = model.predict_proba(img_flat)[0]
            
            st.write(f"## AI Prediction: **{prediction}**")
            st.progress(float(probs[prediction])) # Show confidence

            # --- 5. REWARD & PUNISHMENT SYSTEM ---
            st.divider()
            st.subheader("Feedback Loop")
            st.write("Was the AI correct? Your feedback helps it learn.")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Correct (Reward +5)"):
                    # REWARD: Perform partial fit on the correct prediction
                    model.partial_fit(img_flat, [prediction])
                    st.session_state.ai_score += 5
                    st.toast("AI Rewarded!", icon="🍪")
                    st.rerun()

            with col2:
                actual_digit = st.number_input("If wrong, what was the real digit?", 0, 9, step=1)
                if st.button("❌ Wrong (Punish -5)"):
                    # PUNISH: Force the model to learn the specific correct label
                    model.partial_fit(img_flat, [actual_digit])
                    st.session_state.ai_score -= 5
                    st.toast(f"AI Punished! Learning that was a {actual_digit}", icon="⚡")
                    st.rerun()

    except Exception as e:
        st.error(f"Error processing image: {e}")

# --- 6. INSTRUCTIONS ---
st.sidebar.header("How to Train Your AI")
st.sidebar.write("""
- **Reward**: Clicking 'Correct' reinforces the patterns the AI saw.
- **Punishment**: Providing the correct digit and clicking 'Wrong' uses **Backpropagation** to adjust the neural weights.
""")


[Image of neural network backpropagation diagram]    
