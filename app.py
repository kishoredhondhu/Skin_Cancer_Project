import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import os
import gdown
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import base64
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide"
)

# Create necessary directories
os.makedirs("model_files", exist_ok=True)

# Utility functions
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input"""
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def generate_gradcam(model, img_array, layer_name='conv5_block3_out'):
    """Generate Grad-CAM visualization"""
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    return heatmap

def apply_heatmap(image, heatmap, alpha=0.6):
    """Apply heatmap overlay on image"""
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    img_array = np.array(image)
    colormap = cm.get_cmap("jet")
    heatmap = colormap(heatmap)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap, alpha, 0)
    return Image.fromarray(overlay)

def enhance_image(image):
    """Apply image enhancement techniques"""
    enhancer = ImageEnhance.Contrast(image)
    enhanced = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(1.3)
    return enhanced

def get_image_download_link(img, filename, text):
    """Generate a link to download an image"""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Model loading with fallback strategies
@st.cache_resource
def load_detection_model():
    model_dir = './model_files'
    model_path = os.path.join(model_dir, 'final_model.keras')
    
    # Check if model file exists
    model_file_exists = os.path.exists(model_path) and os.path.getsize(model_path) > 10000
    
    if not model_file_exists:
        st.info("Model files not found. Downloading model...")
        
        # Download options outside the cached function
        download_option = "Automatic download"
        
        if download_option == "Automatic download":
            try:
                file_id = '1OOefzvwsvstYOXpt325S-py2Vc5KPKM4'
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, model_path, quiet=False)
                
                if os.path.exists(model_path) and os.path.getsize(model_path) > 10000:
                    st.success(f"Downloaded model: {os.path.getsize(model_path)} bytes")
                else:
                    st.error("Download failed or file is too small.")
            except Exception as e:
                st.error(f"Download error: {str(e)}")
    
    # Attempt to load model
    try:
        if os.path.exists(model_path) and os.path.getsize(model_path) > 10000:
            model = load_model(model_path, compile=False)
            return model
        else:
            raise Exception("Model file not found or too small")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Creating a simplified model for demonstration purposes.")
        
        # Create a basic model as fallback
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', name='conv5_block3_out')(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
        model = tf.keras.Model(inputs, outputs)
        
        st.warning("Using fallback model - predictions will not be accurate")
        return model

# Main app function
def main():
    # App title
    st.title("Skin Cancer Detection")
    
    # Add tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Detection", "Information", "About"])
    
    with tab1:
        st.header("Skin Lesion Detection")
        st.write("Upload a dermoscopic image to detect if it's benign or malignant.")
        
        # File uploader with additional options
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a dermoscopic image...", 
                type=["jpg", "jpeg", "png"]
            )
            
        with col2:
            use_enhancement = st.checkbox("Use image enhancement", value=True)
        
        # Process uploaded image
        if uploaded_file is not None:
            # Generate a unique ID for the analysis
            analysis_id = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Display original image
            image = Image.open(uploaded_file)
            
            # Apply enhancement if selected
            if use_enhancement:
                enhanced_image = enhance_image(image)
            else:
                enhanced_image = image
            
            # Display images
            display_cols = st.columns(3)
            
            with display_cols[0]:
                st.subheader("Original Image")
                st.image(image, width=224)
            
            # Load model and make prediction
            with st.spinner("Analyzing image..."):
                try:
                    # Get model
                    model = load_detection_model()
                    
                    # Preprocess image for model input
                    img_array = preprocess_image(enhanced_image)
                    
                    # Make prediction
                    prediction = model.predict(img_array)
                    
                    # Get class prediction and confidence
                    class_names = ['Benign', 'Malignant']
                    predicted_class = np.argmax(prediction[0])
                    predicted_label = class_names[predicted_class]
                    confidence = float(prediction[0][predicted_class]) * 100
                    
                    # Generate Grad-CAM
                    try:
                        heatmap = generate_gradcam(model, img_array)
                        heatmap_img = apply_heatmap(enhanced_image, heatmap)
                        
                        with display_cols[1]:
                            st.subheader("Attention Map")
                            st.image(heatmap_img, width=224)
                            st.caption("Areas the model focused on")
                    except Exception as e:
                        st.warning(f"Could not generate attention map: {str(e)}")
                        heatmap_img = None
                    
                    # Display results
                    with display_cols[2]:
                        st.subheader("Prediction")
                        
                        # Show result with appropriate color
                        if predicted_label == "Benign":
                            st.success(f"**Result: {predicted_label}**")
                        else:
                            st.error(f"**Result: {predicted_label}**")
                        
                        st.write(f"**Confidence: {confidence:.2f}%**")
                        
                        # Visualization of probabilities
                        st.write("**Probability Distribution:**")
                        probs = prediction[0] * 100
                        
                        # Create probability bars
                        fig, ax = plt.subplots(figsize=(5, 2))
                        bars = ax.barh(['Benign', 'Malignant'], probs, color=['green', 'red'])
                        ax.set_xlim(0, 100)
                        ax.set_xlabel('Probability (%)')
                        ax.bar_label(bars, fmt='%.1f%%')
                        st.pyplot(fig)
                    
                    # Download options
                    st.subheader("Download Options")
                    download_cols = st.columns(2)
                    
                    with download_cols[0]:
                        st.markdown(get_image_download_link(image, f"original_{analysis_id}.jpg", "Download Original Image"), unsafe_allow_html=True)
                    
                    with download_cols[1]:
                        if heatmap_img:
                            st.markdown(get_image_download_link(heatmap_img, f"analysis_{analysis_id}.jpg", "Download Analysis Image"), unsafe_allow_html=True)
                    
                    # Display disclaimer
                    st.warning("**Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Please try with a different image.")
    
    # Information Tab
    with tab2:
        st.header("Educational Information")
        
        info_tabs = st.tabs(["About Skin Cancer", "ABCDE Rule", "When to See a Doctor"])
        
        # About Skin Cancer
        with info_tabs[0]:
            st.subheader("About Skin Cancer")
            
            st.markdown("""
            ### What is Skin Cancer?
            
            Skin cancer is the abnormal growth of skin cells, most often developed on skin exposed to the sun. 
            It can also occur in areas of the skin not ordinarily exposed to sunlight.
            
            ### Common Types:
            
            - **Basal Cell Carcinoma**: Most common type, rarely spreads
            - **Squamous Cell Carcinoma**: Second most common type, can spread if untreated
            - **Melanoma**: Most dangerous type, can spread to other parts of the body
            
            ### Risk Factors:
            
            - Excessive sun exposure
            - History of sunburns
            - Fair skin
            - Family or personal history of skin cancer
            - Weakened immune system
            """)
        
        # ABCDE Rule Tab
        with info_tabs[1]:
            st.subheader("ABCDE Rule for Melanoma Detection")
            
            st.markdown("""
            The ABCDE rule is a simple guide to help identify potential melanomas:
            
            - **A - Asymmetry**: One half of the mole doesn't match the other half.
            - **B - Border**: The edges are irregular, ragged, notched, or blurred.
            - **C - Color**: The color is not the same throughout and may include different shades.
            - **D - Diameter**: The mole is larger than 6 millimeters across.
            - **E - Evolving**: The mole is changing in size, shape, or color over time.
            """)
        
        # When to See a Doctor Tab
        with info_tabs[2]:
            st.subheader("When to See a Doctor")
            
            st.markdown("""
            ### Consult a dermatologist if you notice:
            
            - A new spot on your skin
            - A spot that differs from other spots on your skin
            - A spot that changes in size, shape, or color
            - A spot that itches, bleeds, or doesn't heal
            - A mole that matches any of the ABCDE criteria
            
            ### Regular Check-ups
            
            - People with no history of skin cancer: Annual skin examination
            - People with a history of skin cancer: More frequent check-ups
            - People with many moles: Regular monitoring
            
            ### Remember
            
            Early detection is crucial for successful treatment of skin cancer.
            """)
    
    # About Tab
    with tab3:
        st.header("About This Application")
        
        st.markdown("""
        ### Purpose
        
        This application uses deep learning to analyze dermoscopic images and detect potential skin cancer. 
        The model classifies skin lesions as either benign or malignant based on visual patterns.
        
        ### How It Works
        
        1. You upload a dermoscopic image
        2. The AI model processes and analyzes the image
        3. Grad-CAM visualization shows areas of interest
        4. Classification results display with confidence scores
        
        ### Limitations
        
        - This is a demonstration tool and should not be used for clinical diagnosis
        - The model was trained on a specific dataset and may not generalize to all types of skin lesions
        - Image quality significantly affects performance
        
        ### Technologies Used
        
        - TensorFlow and Keras for deep learning
        - Streamlit for the web interface
        - Grad-CAM for visualization
        """)

if __name__ == "__main__":
    main()