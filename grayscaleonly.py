import cv2
import numpy as np
import os
import math
# dataset dapat di akses pada https://github.com/williamchand/Image-Digital-Processing
# program dapat langsung dicoba pada LiveDetection.py
	
#mengekseksekusi face detection
def grayscaleonly():
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
					bebasPre = cv2.imread('dataset/'+name+'/'+lux+'/'+img)
					small_frame = cv2.resize(bebasPre, (0, 0), fx=scale, fy=scale)
					gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
					count += 1
					cv2.imwrite("clahe/"+name+"/"+lux+"/%dori.jpg"% count,gray)
grayscaleonly()