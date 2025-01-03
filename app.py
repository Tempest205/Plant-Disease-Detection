import os
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import csv
import io

#For image capture
import time
import threading
import subprocess


# Class Names
class_name = ['Bacterial Spot__Blight',
              'Not Recognized',
              'Tobacco Caterpillar',
              'Tomato Healthy',
              'Tomato Leaf Curl',
              'Tomato Leaf Miner Flies']

# Load trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('NTomato_model_best.keras')

        st.write("Model loaded successfully")

        print(model.summary())
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to capture images using the CSI camera
def capture_image(folder="captured_images", interval=10, num_images=1):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(num_images):
        image_path = os.path.join(folder, f"image_{i+1}.jpg")
        subprocess.run([
            "libcamera-still",
            "-o", image_path,
            "--awb", "auto",  # Adjust white balance
            "--brightness", "0.0",  # Neutral brightness
            "--contrast", "1.0",  # Neutral contrast
            "--saturation", "0.0",  # Neutral saturation
            "--timeout", "1000"  # Capture timeout in milliseconds
        ])
        time.sleep(interval)
    return image_path

    # Initialize session state for the captured image
if "captured_image" not in st.session_state:
    st.session_state.captured_image = None


# Define function for prediction
# Preprocess the image using keras.preprocessing
def preprocess_image(image):
    if isinstance(image, Image.Image):  # Confirming it's a valid PIL Image
        # Convert to RGB to ensure 3-channel input as the model has been trained
        image = image.convert("RGB")
        # Resize with LANCZOS filter to match model input size
        image = image.resize((512, 512), Image.LANCZOS)
        # Convert to numpy array and normalize pixel values to [0, 1]
        input_arr = np.array(image, dtype=np.float32)   # Normalize directly
        return np.expand_dims(input_arr, axis=0)  # Add batch dimension
    raise ValueError("Uploaded file is not a valid image")

# Prediction function
def predict(image_path, model):
    try:
        # Load the image
        image = Image.open(image_path)

        # Preprocess the image
        input_arr = preprocess_image(image)
        print("Image array shape after preprocessing:", input_arr.shape)

        # Check if input array has meaningful values
        print("Sample values from preprocessed image array:", input_arr[0][0][0])  # Log sample pixel values

        # Make prediction
        prediction_probs = model.predict(input_arr)
        result_index = int(tf.argmax(prediction_probs, axis=1).numpy()[0])
        print("Prediction probabilities:", prediction_probs)  # Log prediction probabilities
        print("Predicted class index:", result_index)  # Log predicted class index

        return result_index, prediction_probs
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Function to predict on multiple images in a folder
def predict_images_in_folder(folder_path, class_name, model):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        st.error("No images found in the specified folder.")
        return
    predictions = []

    for file_name in image_files:
        image_path = os.path.join(folder_path, file_name)

        # Load and process the image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper display

        # Convert image to PIL and predict
        pil_image = Image.fromarray(img_rgb)
        result_index, prediction_probs = predict(image_path, model)

        if result_index is not None:
            predicted_class = class_name[result_index]
            predictions.append((file_name, predicted_class, img_rgb))

    # Display images and predictions
    st.write("Predictions:")
    for file_name, predicted_class, img_rgb in predictions:
        st.image(img_rgb, caption=f"Original Image: {file_name}, Predicted Disease: {predicted_class}",
                 use_column_width=True)


# Recommendations for tomato diseases
recommendations = {
    "Tomato Leaf Miner Flies": {
        "Scientific Name": "Tuta absoluta/Liriomyza spp.",
        "Symptoms": [
            "Serpentine pale grey or white lines on leaves.",
            "Yellowing and wilting of affected leaves.",
            "Damaged leaves may drop prematurely (defoliation)."
        ],
        "Image": "Leaf Miner Flies.jpg",  # Leaf miner image path
        "Actions": [
            "Prune and destroy infested leaves which may still harbor the larvae.",
            "Encourage natural predators like parasitic wasps or predatory mirid bugs which disrupt the growth of the larvae in leaves.",
            "Inter-cropping the tomato with suitable wild plants such as sesame to enhance mirid activity.",
            "Place pheromone traps or yellow sticky traps to monitor and reduce adult population.",
            "Use insecticides such as Indoking 300SC(*Dosage: 125ml in 1000 L of water per hectares (2.5ml in 20L of water)*) or Benocarb 100SC(*Dosage: 10ml/20L 0r 250ml/1000L*) which contains Indoxicarb and Emamectin Benzoate to fight the leaf miner flies.",
            "Apply neem oil (Azadirachtin) or biological controls such as Bacillus thuringiensis."
        ]
    },
    "Tomato Leaf Curl": {
        "Scientific Name": "Tomato Leaf Curl Virus (TLCV)",
        "Symptoms": [
            "Upward curling of leaves.",
            "Stunted plant growth.",
            "Yellowing of leaf margins."
        ],
        "Image": "Leaf Curl.jpg",  # Tomato leaf curl image path
        "Actions": [
            "Control whiteflies, which transmit the virus, using sticky traps.",
            "Remove and destroy infected plants immediately.",
            "Use virus-resistant tomato varieties such as Sophya F1 for planting.",
            "Introduce biological insecticides containing Beauveria bassiana, a naturally occurring fungus. This fungus infects and controls whiteflies, offering an eco-friendly alternative to chemical interventions.",
            "Practice crop rotation to minimize viral buildup.",
            "Use Neem oil,at a rate of 5ml per liter, which acts as a repellent and disrupt the life cycle of whiteflies.",
            "Consider using insecticides such as Presento® 200SP(*Dosage:0.25 g per litre of water*) or Acetak 200SL(*Dosage:500ml/ha/1000L of water, 5-10ml/20L*) that contain Acetamiprid which is effective against whiteflies."
        ]
    },
    "Tobacco Caterpillar": {
        "Scientific Name": "Spodoptera litura",
        "Symptoms": [
            "Large, irregular holes in leaves.",
            "Chewed fruits and stems.",
            "Presence of caterpillars on plants."
        ],
        "Image": "Caterpillar.jpg",  # Tobacco Caterpillar image path
        "Actions": [
            "Handpick caterpillars and destroy them.",
            "Encourage natural predators like birds or beneficial insects.",
            "Use biopesticides such as *Bacillus thuringiensis* or neem oil.",
            "Consider using insecticides such as Emmaron 30SC(*Dosage:15ml/20L of water*) that contain Lufenuron and Emmamectin benzoate which are effective against caterpillars.",
            "Apply pheromone traps to monitor adult moth population."
        ]
    },
    "Bacterial Spot__Blight": {
        "Scientific Name": [
            "Xanthomonas spp. (Bacterial Spot)",
            "Phytophthora infestans (Late Blight)",
            "Alternaria solani (Early Blight)"
        ],
        "Symptoms": [
            "Dark, water-soaked spots on leaves and stems (bacterial spot).",
            "Black or brown lesions on older leaves (early blight).",
            "Greasy-looking, gray-green lesions on leaves (late blight)."
        ],
        "Image": "Bacterial spot__Blight.jpg",  # Bacterial spot image path
        "Actions": [
            "Remove and destroy infected plant parts immediately.",
            "Avoid overhead watering to reduce moisture on foliage.",
            "Apply copper-based fungicides such as Green Cop® 500WP and Trinity Gold® 452WP(*Dosage:2.5-3.0 g per litre of water*) for bacterial spot.",
            "Use fungicides such ABSOLUTE STAR 400 SC(*at 10ml in 20L of water*) and TOWER EXTREME 680 WG(*at 50g in 20L of water*) for early and late blight.",
            "Plant disease-resistant varieties and ensure good air circulation."
        ]
    }
}

# Function to display recommendations
def display_recommendation(predicted_class):
    data = recommendations.get(predicted_class, {})
    if not data:
        st.error("No recommendations found for the selected disease.")
        return

    st.header(predicted_class)
    st.subheader(f"Scientific Name: {data.get('Scientific Name')}")

    # Layout for Symptoms and Image
    col1, col2 = st.columns([2, 1])  # Adjust column proportions if needed

    with col1:
        st.subheader("Symptoms")
        for symptom in data.get("Symptoms", []):
            st.write(f"- {symptom}")

    with col2:
        # Resize the image to 512x512 and display
        image_path = data.get("Image")
        if image_path:
            try:
                image = Image.open(image_path)
                resized_image = image.resize((512, 512))  # Resize to (512, 512)
                st.image(resized_image, caption=f"Example of {predicted_class} symptoms", use_container_width=True)
            except FileNotFoundError:
                st.error("Image file not found.")
        else:
            st.error("No image available for this disease.")

    st.subheader("Actions")
    for action in data.get("Actions", []):
        st.write(f"- {action}")

# Create the Streamlit app

# Sidebar
st.sidebar.title('Dashboard')
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'About', 'Disease Recognition'])

# Home Page
if app_mode == 'Home':
    st.title('TOMATO DISEASE CLASSIFICATION')
    image_path = 'home page image.jpg'
    st.image(image_path, use_container_width=True)
    st.markdown('''Our mission is to help farmers, gardeners, and plant enthusiasts quickly identify tomato diseases for healthier and more resilient crops. Just upload a plant image, and our advanced algorithms will analyze it to detect any signs of disease!

### 🌱 How It Works
1. *Upload an Image*: Go to the *Disease Recognition* page and upload an image of your plant.
2. *Image Analysis*: Our powerful machine learning model will scan the image for potential diseases.
3. *View Results*: Get a diagnosis with actionable recommendations to help protect your tomato plants.

### 🌟 Why Choose Us?
- *🌍 Accuracy*: Harnesses cutting-edge AI for precise disease detection.
- *💡 User-Friendly*: Simple, intuitive interface for everyone.
- *⚡ Fast and Efficient*: Receive results in seconds to support quick decision-making.

### 🚀 Get Started
Head to the *Disease Recognition* page in the sidebar, upload an image, and see our Tomato Disease Recognition System in action!

---

### 🧑‍🌾 About Us
Visit the *About* page to learn more about our project, the team behind it, and our commitment to promoting healthier plants.

---
    ''')

# About Page
elif app_mode == 'About':
    st.header('About')
    st.markdown('''
    ### Welcome to Our Tomato Disease Recognition System 🌱

    An innovative solution tailored to address the critical challenges faced by farmers in Kenya. Agriculture is the backbone of Kenya's economy, and tomatoes are one of the most widely cultivated crops. However, the productivity of tomato farming is under constant threat due to devastating plant diseases, which lead to significant economic losses and food insecurity.

    #### Why This Matters
    Farmers often struggle with timely and accurate disease diagnosis, which is crucial for effective treatment and prevention. Traditional methods of diagnosis are not only time-consuming but may also require expert knowledge, which is not always accessible to small-scale farmers. This web app harnesses the power of advanced machine learning and computer vision to bridge that gap, providing farmers with an easy-to-use tool for fast and accurate disease detection.

    #### Our Team 👩‍💻👨‍💻
    This project is a result of a collaborative effort between three dedicated team members and our esteemed supervisor. Together, we have combined expertise in machine learning, computer vision, and agricultural research to develop a system that is practical, user-friendly, and impactful.

    #### Our Mission 🌍
    By empowering farmers with cutting-edge technology, we aim to enhance tomato crop health, improve yields and contribute to the agricultural sustainability of Kenya. With this tool, we hope to make disease detection more accessible and actionable, ensuring a brighter future for Kenyan farmers and their communities.

    Thank you for being a part of this journey toward smarter, sustainable farming!
    ''')


# Prediction Page

elif app_mode == 'Disease Recognition':
    import time
    st.header('Disease Recognition')

    input_option = st.radio("Choose an input method:",
        options=["", "Upload from Device", "Take Photo"],
        index=0,
        format_func=lambda x: "select an option" if x == "" else x
    )


    # JavaScript for playing notification sound
    def play_sound(file_path):
        st.markdown(
            f"""
            <audio autoplay>
                <source src="{file_path}" type="audio/mpeg">
                Your browser does not support the audio element.
            </audio>
            """,
            unsafe_allow_html=True,
        )

    if input_option == "Upload from Device":
        st.info("This option will trigger the CSI camera to capture 10 images.")

        # Button to trigger the image capture
        if st.button('Start Image Capture'):
            with st.spinner('Capturing image...'):
                captured_image_path = capture_image()
                st.session_state.captured_image = captured_image_path
            st.success('Image captured!')

        # Display captured image 
        if st.session_state.captured_image:
            image_path = st.session_state.captured_image
            if st.button('Show Captured Image'):
                image = Image.open(image_path)
                st.image(image, caption="Captured Image", use_container_width=True)


            # Predict Button
            if st.button('Predict'):
                with st.spinner('Please wait...'):
                    model = load_model()  # Load the model
                    if model:
                        result_index, prediction_probs = predict(image_path, model)
                        if result_index is not None:
                            predicted_class = class_name[result_index]
                            st.success(f'Model is predicting it is {predicted_class}')

                            # Play sound if the prediction is not "Healthy"
                            if predicted_class != "Healthy":
                                play_sound("Nextel.mp3")

                            progress = st.progress(0)
                            for i in range(100):
                                time.sleep(0.05)  # Simulate some work
                                progress.progress(i + 1)
                            if predicted_class in recommendations:
                                 # Display the recommendation for the predicted class
                                display_recommendation(predicted_class)

    elif input_option == "Take Photo":
        # Allow user to take an image using device camera
        st.info("Please ensure your browser allows camera access to use this feature.")

        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            #open image taken
            image = Image.open(io.BytesIO(camera_image.getvalue()))
            st.image(image, caption="Taken image", use_container_width=True)
            st.write('')
        else:
            st.error("Please take a picture first.")
        # Predict Button
        if st.button('Predict'):
            if camera_image is not None:
                with st.spinner('Please wait...'):
                    model = load_model()  # Load the model
                    if model:
                        result_index, prediction_probs = predict(camera_image, model)
                        if result_index is not None:

                            predicted_class = class_name[result_index]

                            st.success(f'Model is predicting it’s  {predicted_class}')


                            # Play sound if the prediction is not "Healthy"
                            if predicted_class != "Healthy":
                                play_sound("mixkit-achievement-bell-600.wav")


                            progress = st.progress(0)
                            for i in range(100):
                                time.sleep(0.05)  # Simulate some work
                            progress.progress(i + 1)


                            if predicted_class in recommendations:
                                # Display the recommendation for the predicted class
                                display_recommendation(predicted_class)
