#live camera
# dataset dapat di akses pada https://github.com/williamchand/Image-Digital-Processing
# program dapat langsung dicoba pada LiveDetection.py
import cv2
import numpy as np
import os
import math
def sigmoid(x):
	return math.tanh(x)

class NN:
	def __init__(self, ni, nh, no):
		# number of input, hidden, and output nodes
		self.ni = ni + 1 # +1 for bias node
		self.nh = nh
		self.no = no

        # activations for nodes
		self.ai = [1.0]*self.ni
		self.ah = [1.0]*self.nh
		self.ao = [1.0]*self.no
		
        # create weights
		self.wi = [[-1.377568869748218, 1.6578021203632767, -9.294314914819964, -0.7067922386326493, 0.20566090933378647, 0.8538254456389698, -2.907688522069794, 12.869523847058842, 0.8367742195247228, -0.9304633521986807, 0.7725156677011812], [1.0194745535390852, 0.8531904434830554, -0.4886482604543607, 0.06244465704015051, 0.09530672474694044, 0.7856539314009778, -0.692415539122037, 7.2579069689281885, 1.4577931526093355, -0.17843547476331523, 0.17869820511464093], [0.0925507297531801, 0.06612883234624231, -0.2138498236085642, -0.20377252134190116, -0.11653192833945361, 0.06726460202170066, 0.10875884759451289, 0.18811161893939693, 0.16153019246227213, 0.008186924140624828, 0.04911770845608241]]
		self.wo = [[-0.5698741643493227], [0.2867531629518865], [-2.231346642094346], [-0.046499185250759], [0.4643281290042934], [1.5765553730478832], [-0.24541751864289352], [-1.5226159366377552], [-1.0878470015022617], [1.1659086923538433], [-0.4256747206379448]]
        # last change in weights for momentum
		self.ci = [[-0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0], [-0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0], [-0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0]]
		self.co = [[-0.048503434025658966], [0.048503434025658966], [-0.048503434025658966], [-0.048503434025658966], [0.048503434025658966], [0.048503434025658966], [-0.048503434025658966], [0.048503434025658966], [0.048503434025658966], [-0.048503434025658966], [0.048503434025658966]]

	def update(self, inputs):
		if len(inputs) != self.ni-1:
			raise ValueError('wrong number of inputs')

		# input activations
		for i in range(self.ni-1):
			#self.ai[i] = sigmoid(inputs[i])
			self.ai[i] = inputs[i]

		# hidden activations
		for j in range(self.nh):
			sum = 0.0
			for i in range(self.ni):
				sum = sum + self.ai[i] * self.wi[i][j]
			self.ah[j] = sigmoid(sum)

		# output activations
		for k in range(self.no):
			sum = 0.0
			for j in range(self.nh):
				sum = sum + self.ah[j] * self.wo[j][k]
			self.ao[k] = sigmoid(sum)

		return self.ao[:]

	def test(self, patterns):
		errno = 0
		for p in patterns:
			a=self.update(p[0])[0]
			if (a<0.5):
				temp = 0
			else:
				temp = 1
		return(temp)
	
#mengekseksekusi face detection
cascPath = "model/cascade.xml"
faceCascadeCustom = cv2.CascadeClassifier(cascPath)
cam = cv2.VideoCapture(0)
cam.set(3, 1280) # set video widht
cam.set(4, 960) # set video height
scale =0.5
while True:
	ret, img = cam.read()
	small_frame = cv2.resize(img, (0, 0), fx=scale, fy=scale)
	gray = cv2.cvtColor(small_frame,cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(gray)
	faces = faceCascadeCustom.detectMultiScale(
		cl1,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		)
	for x,y,w,h in faces:
		x1 = int(x/scale)
		y1 = int(y/scale)
		x2 = int((x+w)/scale)
		y2 = int((y+h)/scale)
		#cv2.rectangle(bebasPre, (int(x/scale), int(y/scale)), (int((x+w)/scale), int((y+h)/scale)), (0, 255, 0), 6)
		wajah = img[y1:y2,x1:x2]
		sumHue = 0
		sumValue = 0
		#checking hue value
		hsv = cv2.cvtColor(wajah,cv2.COLOR_BGR2HSV)
		hue = cv2.calcHist( [hsv], [0], None, [256], [0, 256] )
		value = cv2.calcHist( [hsv], [2], None, [256], [0, 256] )
		for k in range (0,20):
			sumHue = hue[k]+sumHue
		for k in range (200,255):
			sumValue = value[k]+sumValue
		neuralin = [[[sumHue,sumValue]]]
		# create a network with two input, two hidden, and one output nodes
		n = NN(2, 11, 1)
		# train it with some patterns
		indikatorPos=n.test(neuralin)
		if (indikatorPos==1): #predicted not face HV in NN
			cv2.rectangle(img, (int(x/scale), int(y/scale)), (int((x+w)/scale), int((y+h)/scale)), (0, 255, 0), 6)
	cv2.imshow('camera',img) 

	k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
	if k == 27:
		break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()