import easyocr
import pyttsx3
import matplotlib.pyplot as plt
import cv2

def is_spelled_correctly(word, reference_dictionary):
    return word.lower() in reference_dictionary

def adjust_bbox_all_text(bboxes, img_shape, expansion_pixels=5):
    min_x = min(bbox[0][0] for bbox in bboxes)
    min_y = min(bbox[0][1] for bbox in bboxes)
    max_x = max(bbox[2][0] for bbox in bboxes)
    max_y = max(bbox[2][1] for bbox in bboxes)

    top_left = max(0, min_x - expansion_pixels), max(0, min_y - expansion_pixels)
    bottom_right = min(img_shape[1], max_x + expansion_pixels), min(img_shape[0], max_y + expansion_pixels)

    return top_left, bottom_right


def detect_misspelled_words(scanned_image_path, reference_dictionary, expansion_pixels=25, line_thickness=30):
    reader = easyocr.Reader(['en'])
    img = cv2.imread(scanned_image_path)
    img_shape = img.shape[:2]
    results = reader.readtext(img)
    reader = easyocr.Reader(['en'], gpu=False)
    text_ = reader.readtext(img)

    bboxes = [result[0] for result in results]
    (top_left, bottom_right) = adjust_bbox_all_text(bboxes, img_shape, expansion_pixels)

    cv2.rectangle(img, tuple(map(int, top_left)), tuple(map(int, bottom_right)), (0, 255, 0), line_thickness)

    detected_text = [result[1] for result in results]
    misspelled_words = [word for word in detected_text if not is_spelled_correctly(word, reference_dictionary)]

    return img, detected_text, misspelled_words


reference_dictionary = ["TEAPOT"]
scanned_image_path = r"C:\Users\ASUS\Downloads\TEAPOT1.png"

text_speech = pyttsx3.init()
text_speech.say("TEAPOT")
text_speech.runAndWait()

result_image, detected_text, misspelled_words = detect_misspelled_words(scanned_image_path, reference_dictionary,
                                                                        expansion_pixels=25, line_thickness=10)

plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.show()

print("Detected Words:", detected_text)
print("Reference Words:", reference_dictionary)
