from deepface import DeepFace
import cv2

def detect_face_sentiment(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Analyze emotion using DeepFace
    analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
    dominant_emotion = analysis[0]['dominant_emotion']
    
    return dominant_emotion

