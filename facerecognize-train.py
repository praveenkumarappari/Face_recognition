import cv2 as cv
import os
import numpy as nmp
from PIL import Image
import pickle as pk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_casc = cv.CascadeClassifier('cascading/data/frontalface_alt2.xml')
identifier = cv.face.LBPHFaceRecognizer_create()

curr_id = 0
lbl_id = {}
y_lbl = []
x_trn = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label1 = os.path.basename(root).replace(" ", "-").lower()
			if not label1 in lbl_id:
				lbl_id[label1] = curr_id
				curr_id += 1
			id_ = lbl_id[label1]			
			grey_image = Image.open(path).convert("L") # grayscale
			size = (550, 550)
			f_image = grey_image.resize(size, Image.ANTIALIAS)
			image_array = nmp.array(f_image, "uint8")
			faces_scale = face_casc.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

			for (x,y,w,h) in faces_scale:
				roi = image_array[y:y+h, x:x+w]
				x_trn.append(roi)
				y_lbl.append(id_)



with open("pk/face-labels.pickle", 'wb') as f:
	pk.dump(lbl_id, f)

identifier.train(x_trn, nmp.array(y_lbl))
identifier.save("identifiers/face-trainner.yml")