import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import nltk

# Download the words corpus if it's not already downloaded
nltk.download('words')

# Read the image
image_path ="D:\\apple.jpg"
img = cv2.imread(image_path)

# Instance text detector
reader = easyocr.Reader(['en'], gpu=False)

# Detect text on image
text_ = reader.readtext(img)

# Create a set of words from the NLTK words corpus for efficient lookup
word_set = set(nltk.corpus.words.words())

# Set threshold for bounding box visualization
threshold = 0.25

# Draw bounding box and text
for bbox, text, score in text_:
    print(f"Detected text: {text}, Score: {score}")

    # Check if the word exists in the NLTK words corpus
    if text.lower() in word_set:
        print(f"The word '{text}' exists in the dictionary.")
        color = (0, 255, 0)  # Green color for words that exist
    else:
        print(f"The word '{text}' does not exist in the dictionary.")
        color = (0, 0, 255)  # Red color for words that don't exist

    # Draw bounding box if score exceeds threshold
    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], color, 5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 5, (255, 0, 0), 2)

# Display the image with bounding boxes
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
