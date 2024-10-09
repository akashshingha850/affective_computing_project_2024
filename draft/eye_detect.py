import cv2

def load_eye_cascade(eye_cascade_path):
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    return eye_cascade

def process_frame(frame, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Draw rectangles around detected eyes

    return frame

def main():
    eye_cascade_path = '/home/jetson/affective_computing/haarcascades/haarcascade_eye.xml'
    eye_cascade = load_eye_cascade(eye_cascade_path)

    # Initialize video capture from RGB camera (update the device index as needed)
    cap = cv2.VideoCapture('/dev/video0')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame to detect eyes
        frame = process_frame(frame, eye_cascade)

        # Display the resulting frame
        cv2.imshow('Eye Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
