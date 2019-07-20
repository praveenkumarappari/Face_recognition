import numpy as nmp
import cv2 as cv
import pickle as pk

face_casc = cv.CascadeClassifier('cascading/data/frontalface_alt2.xml')
eye_casc = cv.CascadeClassifier('cascading/data/eye.xml')
smile_casc = cv.CascadeClassifier('cascading/data/smile.xml')


identifier = cv.face.LBPHFaceRecognizer_create()
identifier.read("./identifiers/face-trainner.yml")

lbl = {"person_name": 1}                          # this is label
with open("pk/face-labels.pickle", 'rb') as f:
	og_lbl = pk.load(f)
	lbl = {v:k for k,v in og_lbl.items()}

capture = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = capture.read()
    gray_scale  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_scale = face_casc.detectMultiScale(gray_scale, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces_scale:
    	print(x,y,w,h)
    	roi_gray = gray_scale[y:y+h, x:x+w] #(ycord_start, ycord_end)
    	roi_color = frame[y:y+h, x:x+w]

    	# recognize? deep learned model predict keras tensorflow pytorch scikit learn
    	id_, conf = identifier.predict(roi_gray)
    	if conf>=4 and conf <= 85:
    		font = cv.FONT_HERSHEY_SIMPLEX
    		name = lbl[id_]
    		color = (255, 255, 255)
    		stroke = 2
    		cv.putText(frame, name, (x,y), font, 1, color, stroke, cv.LINE_AA)

    	image_item = "test_image.png"
    	cv.imwrite(image_item, roi_color)

    	color = (255, 0, 0) #BGR 0-255 
    	stroke = 2
    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    # Display the resulting frame
    cv.imshow('frame',frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
capture.release()
cv.destroyAllWindows()