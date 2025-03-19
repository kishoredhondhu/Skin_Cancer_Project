import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    # Resize image
    img = tf.image.resize(image, target_size)
    # Normalize pixel values
    img = img / 255.0
    # Add batch dimension
    img = tf.expand_dims(img, 0)
    return img

def generate_gradcam(model, img_array, last_conv_layer="conv5_block3_out"):
    """Generate Grad-CAM visualization"""
    # Create a model that maps the input image to activations and predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer).output, model.output]
    )
    
    # Compute gradient of top predicted class with respect to activations
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
        
    # Gradient of the output neuron with respect to the output feature map
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Vector of mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight output feature map with gradient values
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    
    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap.numpy(), (img_array.shape[1], img_array.shape[2]))
    
    return heatmap

def apply_heatmap(image, heatmap):
    """Apply heatmap on image"""
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert image to BGR for OpenCV
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (224, 224))
    
    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    
    # Convert back to RGB
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return superimposed_img