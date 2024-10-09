import cv2
import os
import time

def calculate_fps(frame_counter, start_time):
    end_time = time.time()
    fps = frame_counter / (end_time - start_time)
    return fps, end_time

def initialize_video_capture(device_index=0, width=1280, height=720):
    cap = cv2.VideoCapture(device_index)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def load_cascades(face_cascade_path, eye_cascade_path):
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    return face_cascade, eye_cascade

def process_frame(frame, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            if ey < h // 2 and 0.2 < ew / eh < 1.2:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                # Uncomment to save eye images
                # save_eye_image(roi_color, ex, ey, ew, eh)
    return frame

def main():
    eye_counter = 0

    # Paths to Haar cascade files
    face_cascade_path = '/home/jetson/affective_computing/haarcascades/haarcascade_frontalface_default.xml'
    eye_cascade_path = '/home/jetson/affective_computing/haarcascades/haarcascade_eye.xml'

    # Load the Haar cascade files
    face_cascade, eye_cascade = load_cascades(face_cascade_path, eye_cascade_path)

    # Start video capture from the RGB camera
    cap = initialize_video_capture('/dev/video0')  #/dev/video6 is the RGB camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, face_cascade, eye_cascade)

        cv2.imshow('Face and Eye Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
