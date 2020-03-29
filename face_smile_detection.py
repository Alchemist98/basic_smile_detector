import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[x:x+w,y:y+h]
        roi_color = frame[x:x+w,y:y+h]
        smile = smile_cascade.detectMultiScale(roi_gray,1.7,22)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)       
    return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    if ret is True:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        continue
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
