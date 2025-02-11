import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

# Load the saved model
model = load_model('twilight_model.h5')


def run(image_path):
    img = cv2.imread(image_path)

    # Convert from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(img_rgb)
    plt.show()

    # Resize the image for model input
    resize = tf.image.resize(img_rgb, (256, 256))

    # Display the resized image
    plt.imshow(resize.numpy().astype(int))
    plt.show()

    # Make a prediction
    yhat = model.predict(np.expand_dims(resize / 255, 0))

    # Get the predicted class and confidence
    predicted_class = np.argmax(yhat)  # Index of the highest value
    confidence = np.max(yhat)  # Confidence level of the prediction

    class_names = {
        0: "Arbiters Grounds",
        1: "City in the Sky",
        2: "Forest Temple",
        3: "Goron Mines",
        4: "Hyrule Castle",
        5: "Lakebed Temple",
        6: "Palace of Twilight",
        7: "Snowpeak Ruins",
        8: "Temple of Time"
    }

    predicted_class_name = class_names.get(predicted_class, "Unknown Class")

    # Print the prediction result
    print(f"Predicted Class: {predicted_class_name} with {confidence:.2%} confidence")
    return predicted_class_name, confidence

if __name__ == "__main__":
    image_path = ''
    run(image_path)