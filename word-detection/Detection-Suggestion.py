import cv2
import easyocr
import matplotlib.pyplot as plt
import nltk
from spellchecker import SpellChecker

# Download the words corpus if it's not already downloaded
nltk.download('words')

# Read the image
image_path = "D:\\apple.jpg"
img = cv2.imread(image_path)

# Instance text detector
reader = easyocr.Reader(['en'], gpu=False)

# Detect text on image
text_ = reader.readtext(img)

# Create a set of words from the NLTK words corpus for efficient lookup
word_set = set(nltk.corpus.words.words())

# Initialize SpellChecker
spell = SpellChecker()

# Set threshold for bounding box visualization
threshold = 0.25

# Function to get Levenshtein distance between two words
def levenshtein_distance(word1, word2):
    if len(word1) < len(word2):
        return levenshtein_distance(word2, word1)

    if len(word2) == 0:
        return len(word1)

    previous_row = range(len(word2) + 1)

    for i, c1 in enumerate(word1):
        current_row = [i + 1]
        for j, c2 in enumerate(word2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

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

        # Get suggestions for the misspelled word
        suggestions = spell.candidates(text)
        print(f"Suggestions: {suggestions}")

        # Limit the number of suggestions to display
        num_suggestions = 3  # You can adjust this number as needed

        # Sort suggestions by Levenshtein distance
        sorted_suggestions = sorted(suggestions, key=lambda x: levenshtein_distance(x, text))

        # Get top suggestions
        suggested_words = sorted_suggestions[:num_suggestions]
        print(f"Top {num_suggestions} suggestions: {suggested_words}")

    # Draw bounding box if score exceeds threshold
    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], color, 5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 5, (255, 0, 0), 2)

# Display the image with bounding boxes
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
