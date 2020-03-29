import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

test = cv2.imread('test2.jpg')

gray_img = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img,1.3,5)

for (x,y,w,h) in faces:
    cv2.rectangle(test,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray_img[x:x+w,y:y+h]
    roi_color = test[x:x+w,y:y+h]
    smiles = smile_cascade.detectMultiScale(roi_gray,1.7,22)
    for (sx,sy,sw,sh) in smiles:
        cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)
        
cv2.imshow('Test Img',test)
cv2.waitKey(0)
cv2.destroyAllWindows()