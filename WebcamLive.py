import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Failed to grab frame")
        break

    # Analyze emotions
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    # Get dominant emotion
    dominant_emotion = result[0]['dominant_emotion']
    print("Dominant Emotion:", dominant_emotion)

    # Display the frame with emotion text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Emotion: {dominant_emotion}"
    position = (10, 50)
    font_scale = 1
    color = (0, 255, 0)  # Green color
    thickness = 2

    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Emotion Detector", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
