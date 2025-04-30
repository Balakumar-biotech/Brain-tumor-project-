# Brain Tumor Segmentation Using OpenCV and Deep Learning

# Import necessary libraries
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanIoU
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import unittest

# Constants
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 25

# Data augmentation
def create_data_generators(train_dir, val_dir):
    datagen_args = dict(rescale=1./255,
                        rotation_range=20,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.1,
                        horizontal_flip=True,
                        fill_mode='nearest')

    image_datagen = ImageDataGenerator(**datagen_args)
    mask_datagen = ImageDataGenerator(**datagen_args)

    image_generator = image_datagen.flow_from_directory(
        train_dir, target_size=(IMG_SIZE, IMG_SIZE), class_mode=None, batch_size=BATCH_SIZE, seed=1)

    mask_generator = mask_datagen.flow_from_directory(
        val_dir, target_size=(IMG_SIZE, IMG_SIZE), class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=1)

    train_generator = zip(image_generator, mask_generator)
    return train_generator

# U-Net Model
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate

def build_unet():
    inputs = Input((IMG_SIZE, IMG_SIZE, 1))

    # Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    # Decoder
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Compile and train model
def train_model(train_generator, val_generator):
    model = build_unet()
    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy', MeanIoU(num_classes=2)])
    history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)
    model.save("unet_model.h5")
    return model, history

# Load pretrained U-Net model
def load_unet_model(model_path):
    return load_model(model_path, compile=False)

# Preprocess image
def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict mask
def predict_mask(model, image):
    pred_mask = model.predict(image)
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    pred_mask = np.squeeze(pred_mask) * 255
    return pred_mask

# Post-process mask
def postprocess_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# Overlay mask on original image
def overlay_mask(image, mask):
    color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(image.convert('RGB')), 0.7, color_mask, 0.3, 0)
    return overlay

# Streamlit App
def main():
    st.title("Brain Tumor Segmentation")
    model = load_unet_model("unet_model.h5")

    uploaded_file = st.file_uploader("Upload an MRI Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        processed_img = preprocess_image(image)
        mask = predict_mask(model, processed_img)
        post_mask = postprocess_mask(mask)
        overlay = overlay_mask(image, post_mask)

        st.image(post_mask, caption="Predicted Mask", use_column_width=True)
        st.image(overlay, caption="Overlay Image", use_column_width=True)

        # Download mask
        result = Image.fromarray(post_mask)
        st.download_button("Download Mask", data=result.tobytes(), file_name="mask.png", mime="image/png")

if __name__ == "__main__":
    main()

# Unit Tests
class TestSegmentationPipeline(unittest.TestCase):
    def test_preprocess_image(self):
        img = Image.new('L', (512, 512), color=255)
        processed = preprocess_image(img)
        self.assertEqual(processed.shape, (1, 256, 256, 1))

    def test_model_output_shape(self):
        model = build_unet()
        dummy_input = np.zeros((1, 256, 256, 1))
        output = model.predict(dummy_input)
        self.assertEqual(output.shape, (1, 256, 256, 1))

    def test_postprocess_mask(self):
        dummy_mask = np.zeros((256, 256), dtype=np.uint8)
        dummy_mask[100:150, 100:150] = 255
        cleaned_mask = postprocess_mask(dummy_mask)
        self.assertEqual(cleaned_mask.shape, dummy_mask.shape)
        self.assertTrue(np.max(cleaned_mask) <= 255)

    def test_overlay_mask(self):
        img = Image.new('L', (256, 256), color=128)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[100:150, 100:150] = 255
        overlay = overlay_mask(img, mask)
        self.assertEqual(overlay.shape, (256, 256, 3))

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
