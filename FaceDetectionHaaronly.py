import cv2
import numpy as np
import os
import math
# dataset dapat di akses pada https://github.com/williamchand/Image-Digital-Processing
	
# program dapat langsung dicoba pada LiveDetection.py
#mengekseksekusi face detection
def FaceDetectionHaar():
	scale = 0.5
	name1 = ["aji","dion","targit" ,"udin","ule","william","yoland","lisa","hansen"]
	lux1 = ["70","100","150","250","300"]
	cascPath = "model/cascade.xml"
	faceCascadeCustom = cv2.CascadeClassifier(cascPath)
	for lux in lux1:
		for name in name1:
			if os.path.isdir('dataset/'+name+'/'+lux):
				count = 0
				for img in os.listdir('dataset/'+name+'/'+lux):
					bebasHaar = cv2.imread('dataset/'+name+'/'+lux+'/'+img)
					small_frame = cv2.resize(bebasHaar, (0, 0), fx=scale, fy=scale)
					gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
					faces = faceCascadeCustom.detectMultiScale(
						gray,
						scaleFactor=1.1,
						minNeighbors=5,
						minSize=(30, 30)
						)
					for x,y,w,h in faces:
						x1 = int(x/scale)
						y1 = int(y/scale)
						x2 = int((x+w)/scale)
						y2 = int((y+h)/scale)
						cv2.rectangle(bebasHaar, (int(x/scale), int(y/scale)), (int((x+w)/scale), int((y+h)/scale)), (0, 255, 0), 6)
					count += 1
					cv2.imwrite("eval/"+name+"/"+lux+"/%d.jpg"% count,bebasHaar)
FaceDetectionHaar()