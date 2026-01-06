import streamlit as st
import numpy as np
from PIL import Image
import joblib
import requests
import io

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")
st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image and AI will try to recognize it.")

# Simple model loading with fallback
@st.cache_resource
def load_model():
    try:
        # Try to load pre-trained model
        # Using sklearn's built-in digits dataset
        from sklearn.datasets import load_digits
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split

        digits = load_digits()
        X = digits.images.reshape((len(digits.images), -1)) / 16.0
        y = digits.target
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
      26        model = MLPClassifier(
27            hidden_layer_sizes=(100,),
28            max_iter=100,
29            random_state=42
30        )
31        model.fit(X_train, y_train)
32        return model
33    except Exception as e:
34        st.error(f"Model loading error: {e}")
35        return None
36
37 model = load_model()
38
39 if model is None:
40    st.warning("Could not load model. Using fallback recognition.")
41 else:
42    st.success("Model loaded successfully!")
43
44 # File uploader
45 uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
46
47 if uploaded_file is not None:
48    # Display the uploaded image
49    image = Image.open(uploaded_file)
50    st.image(image, caption='Uploaded Image', use_column_width=True)
53    # Process the image
54    try:
55        # Convert to grayscale and resize to 8x8
56        img_gray = image.convert('L')
57        img_resized = img_gray.resize((8, 8))
58
59        # Convert to numpy array and invert if needed
60        img_array = np.array(img_resized)
61
62        # If background is dark, invert
63        if np.mean(img_array) > 128:
64            img_array = 255 - img_array
65
66        # Normalize like the training data
67        img_array = img_array / 16.0
68        img_flat = img_array.flatten().reshape(1, -1)
69
70        if model is not None:
71            # Make prediction
72            prediction = model.predict(img_flat)[0]
73            st.write(f"## Prediction: **{prediction}**")
74
75            # Show probabilities
76            probs = model.predict_proba(img_flat)[0]
77            st.write("### Probabilities:")
78            for i, prob in enumerate(probs):
79                st.write(f"Digit {i}: {prob:.2%}")
80        else:
81            # Fallback: simple threshold-based recognition
82            st.write("## Using fallback recognition")
83            # Simple heuristic based on pixel intensity
26        model = MLPClassifier(
27            hidden_layer_sizes=(100,),
28            max_iter=100,
29            random_state=42
30        )
31        model.fit(X_train, y_train)
32        return model
33    except Exception as e:
34        st.error(f"Model loading error: {e}")
35        return None
36
37 model = load_model()
38
39 if model is None:
40    st.warning("Could not load model. Using fallback recognition.")
41 else:
42    st.success("Model loaded successfully!")
43
44 # File uploader
45 uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
46
47 if uploaded_file is not None:
48    # Display the uploaded image
49    image = Image.open(uploaded_file)
50    st.image(image, caption='Uploaded Image', use_column_width=True)
51
52 # Process the image
53 try:
54     # Convert to grayscale and resize to 8x8
55     img_gray = image.convert('L')
56     img_resized = img_gray.resize((8, 8))
57
58     # Convert to numpy array and invert if needed
59     img_array = np.array(img_resized)
60
61     # If background is dark, invert
62     if np.mean(img_array) > 128:
63         img_array = 255 - img_array
64
65     # Normalize like the training data
66     img_array = img_array / 16.0
67     img_flat = img_array.flatten().reshape(1, -1)
68
69     if model is not None:
70         # Make prediction
71         prediction = model.predict(img_flat)[0]
72         st.write(f"## Prediction: **{prediction}**")
73
74         # Show probabilities
75         probs = model.predict_proba(img_flat)[0]
76         st.write("### Probabilities:")
77         for i, prob in enumerate(probs):
78             st.write(f"Digit {i}: {prob:.2%}")
79     else:
80         # Fallback: simple threshold-based recognition
81         st.write("## Using fallback recognition")
82         # Simple heuristic based on pixel intensity
83         digit_guess = np.argmax(np.sum(img_array.reshape(8, 8), axis=0)) % 10
84         st.write(f"Estimated digit: **{digit_guess}**")
85
86 except Exception as e:
87     st.error(f"Error processing image: {e}")
88
89 # Instructions
90 st.sidebar.header("Instructions")
91 st.sidebar.write("""
92 1. Upload an image of a handwritten digit (0-9)
93 2. The image will be resized to 8x8 pixels
94 3. AI model will predict the digit
95 4. For best results:
96    - White background
97    - Black digit
98    - Centered digit
99    - Minimal noise
100 """)
