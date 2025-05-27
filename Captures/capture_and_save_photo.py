import cv2 as cv
import os

def capture_and_save_face(label_name, save_dir='/Users/mac/Downloads/content/Pazh-Electronic-computer-vision/DataSet/pictures'):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv.VideoCapture(0)
    print("برای گرفتن عکس، دکمه 'space' رو فشار بده. برای خروج 'q' رو بزن.")
    img_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("خطا در باز کردن دوربین.")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

        cv.imshow('Camera', frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord(' '):
            if len(faces) == 0:
                print("هیچ چهره‌ای یافت نشد، عکس ذخیره نشد.")
                continue
            for i, (x, y, w, h) in enumerate(faces):
                face_img = frame[y:y+h, x:x+w]
                img_name = f"{label_name}_{img_count}.jpg"
                img_path = os.path.join(save_dir, img_name)
                cv.imwrite(img_path, face_img)
                print(f"چهره ذخیره شد: {img_path}")
                img_count += 1

        elif key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

label = input("نام فرد را وارد کنید: ")
capture_and_save_face(label)
