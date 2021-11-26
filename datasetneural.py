import cv2
import numpy as np
import os
import math
# dataset dapat di akses pada https://github.com/williamchand/Image-Digital-Processing
# program dapat langsung dicoba pada LiveDetection.py
#membuat dataset training Neural Networks
def makeDataset():
	name1 = ["aji","dion","targit" ,"udin","ule","william","yoland","lisa","hansen"]
	lux1 = ["70","100","150","250","300"]
	numberFace = 0
	numberDetected =0
	numberFalsePositive = 0
	numberFalseNegative = 0
	text_file = open("data.csv","w")
	text_file.write("Hue,Value,Output\n")
	limitX1 = 400
	limitX2 = 800
	limitY1 = 300
	limitY2 = 600
	count=1
	scale = 0.5
	cascPath = "model/cascade.xml"
	faceCascadeCustom = cv2.CascadeClassifier(cascPath)
	for name in name1:
		for lux in lux1:
			if os.path.isdir('dataset/'+name+'/'+lux):
				# skip directories
				for img in os.listdir('dataset/'+name+'/'+lux):
					bebas=cv2.imread('dataset/'+name+'/'+lux+'/'+img)
					small_frame = cv2.resize(bebas, (0, 0), fx=scale, fy=scale)
					gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
					clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
					cl1 = clahe.apply(gray)
					faces = faceCascadeCustom.detectMultiScale(cl1, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
					indikator = 0
					for x,y,w,h in faces:
						x1 = int(x/scale)
						y1 = int(y/scale)
						x2 = int((x+w)/scale)
						y2 = int((y+h)/scale)
						wajah = bebas[y1:y2,x1:x2]
						sumHue = 0
						sumValue = 0
						hsv = cv2.cvtColor(wajah,cv2.COLOR_BGR2HSV)
						hue = cv2.calcHist( [hsv], [0], None, [256], [0, 256] )
						#saturation = cv2.calcHist( [hsv], [0], None, [256], [0, 256] )
						value = cv2.calcHist( [hsv], [2], None, [256], [0, 256] )
						for k in range (0,20):
							sumHue = hue[k]+sumHue
						for k in range (200,255):
							sumValue = value[k]+sumValue
						#asumsi wajah selalu ditengah
						if x1>limitX1 and x2<limitX2 and y1>limitY1 and y2<limitY2:
							indikator = 1
						else:
							indikator = 0
						text_file.write("%d,%d,%d\n" % (sumHue, sumValue, indikator))
	text_file.close()
makeDataset()