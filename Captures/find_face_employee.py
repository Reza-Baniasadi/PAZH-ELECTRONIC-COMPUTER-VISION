import cv2 as cv
import sys
sys.path.append('/Users/mac/Downloads/content/Pazh-Electronic-computer-vision/Model')
from train import recognize_face

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

def lookEmployee():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("خطا در خواندن فریم.")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            try:
                name = recognize_face(face_img)
                color = (0, 255, 0)
                label = name
            except ValueError as e:
                print(e)
                color = (0, 0, 255)
                label = "ناشناس"

            cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv.imshow('Face Recognition', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    lookEmployee()
