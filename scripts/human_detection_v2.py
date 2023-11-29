#!/usr/bin/env python
from __future__ import print_function

import sys
import rospy
import cv2 as cv
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import time
import rospkg
import copy
import numpy as np
from threading import Thread
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

class QueueFPS(queue.Queue):
	def __init__(self):
		queue.Queue.__init__(self)
		self.startTime = 0
		self.counter = 0

	def put(self, v):
		queue.Queue.put(self, v)
		self.counter += 1
		if self.counter == 1:
			self.startTime = time.time()

#Class for detecting an human by using Yolov3-tiny cfg
class HumanDetetcion:
	
	def __init__(self):
		
		#Attributes
		self.bridge = CvBridge()

		self.asyncN = 0 # It stands for how many frames you want to process
		# 0 -> No asyncronus frame process
		# N -> N frames processed syncronusly

		#general_info
		self.IMG_HEIGHT = 480
		self.IMG_WIDTH = 640
		self.RECTANGLE_COLOR = (255,0,0) #BGR
		self.confidence_treshold = .6
		self.process = True

		#Queue for incoming frames
		self.framesQueue = QueueFPS()
		#Queue for predictions
		self.predictionsQueue = QueueFPS()
		#Queue for frames porcessed
		self.processedFramesQueue = queue.Queue()
		
		#Camera image subscriber to the left side
		self.image_sub = rospy.Subscriber("/spot/camera/back/image", Image, self.image_callback)
		#self.camera_param_sub = rospy.Subscriber("/spot/camera/back/camera_info", CameraInfo, self.camera_info_callback)

        #Set the absolute path of the 
		rospack = rospkg.RosPack() 
		
		self.rate = rospy.Rate(100)

		self.K_camera = None
		self.P_camera = None
		
		#Path of the package
		pkg_path = rospack.get_path('spot_human_detetction')
		config_svg_path = pkg_path + '/yolo/yolo-tiny-tabi.cfg'   
		weights_path = pkg_path + '/yolo/yolo-tiny-tabi.weights'
		names_path = pkg_path + "/yolo/tabi.names"
        
        #Load the neural network from Darknet
		self.net = cv.dnn.readNet(config_svg_path, weights_path, 'darknet')
		
		self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
		self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
		
		self.layerNames = self.net.getLayerNames()
		self.lastLayerId = self.net.getLayerId(self.layerNames[-1])
		self.lastLayer = self.net.getLayer(self.lastLayerId)
		self.outNames = self.net.getUnconnectedOutLayersNames()
		
        # Load names of classes and get random colors
		self.classes = open(names_path).read().strip().split('\n')
		
		# determine the output layer
		self.ln = self.net.getLayerNames()
		self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
		
		self.processTimer = rospy.Timer(rospy.Duration(0.0035), self.processingThread)
		self.postProcessTimer = rospy.Timer(rospy.Duration(0.001), self.postProcessThread)

		#cv.namedWindow('Camera')
		cv.namedWindow('Human Detection')
		
		print("Ready...")


	def camera_info_callback(self, data):

		self.K_camera = data.K
		self.P_camera = data.P

		self.camera_param_sub.shutdown()

	#Callback function of camera image subscription	
	def image_callback(self, data):		
		#try to get the image by the format cv::Mat()
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")			
			#When a new frame comes is stored in the queue
			self.framesQueue.put(cv_image)	
		except CvBridgeError as e:
			print(e)

	"""
		Thread body for process frames
	"""
	def processingThread(self, data):
		#print("Pprocess running")
		futureOutputs = []
		#while(self.process):
			
		new_frame = None
		try:
			# Retrive the next frame 
			new_frame = self.framesQueue.get_nowait()
			if self.asyncN:
				if(len(futureOutputs) == self.asyncN):
					new_frame = None
			else:
				# Skip the rest of frames
				self.framesQueue.queue.clear()
		except queue.Empty:				
			pass
	
		if not new_frame is None:
			prev_time = time.time()
			#note!: For yolo swapRB always true
			# Create a 4D blob from the new_frame
			input_blob = cv.dnn.blobFromImage(new_frame, size=(self.IMG_WIDTH, self.IMG_HEIGHT), swapRB=True, ddepth=cv.CV_8U)

			self.processedFramesQueue.put(new_frame)

			# Run model			
			self.net.setInput(input_blob, scalefactor=0.00392, mean=[0, 0, 0]) # scalefactor for yolo: 0.00392

			if self.asyncN:
				futureOutputs.append(self.net.forwardAsync())
			else:
				outs = self.net.forward(self.outNames)
				self.predictionsQueue.put(copy.deepcopy(outs))
		
			while futureOutputs and futureOutputs[0].wait_for(0):
				out = futureOutputs[0].get()
				self.predictionsQueue.put(copy.deepcopy([out]))

				del futureOutputs[0]	

			delta_t = (time.time() - prev_time)
			print("time for prediction: ", delta_t)		
	
	"""
		Thread for postprocess the predictions
	"""
	def postProcessThread(self, data):
		#print("Post process running")
		classIds = []
		confidences = []
		boxes = []

		outs = None

		#while(self.process):

		classIds.clear()
		confidences.clear()
		boxes.clear()

		# Retrive predictions and relative frame
		try:
			outs = self.predictionsQueue.get_nowait()
			frame = self.processedFramesQueue.get_nowait()
		except queue.Empty:
			pass

		box_scale_w = self.IMG_WIDTH
		box_scale_h = self.IMG_HEIGHT

		if outs is None:
			#continue
			return 
		else:			
			# Network produces output blob with a shape NxC where N is a number of
			# detected objects and C is a number of classes + 4 where the first 4
			# numbers are [center_x, center_y, width, height]
			for out in outs:		
				for detection in out:
					scores = detection[4:]
					scores = np.delete(scores, 0)
					classId = np.argmax(scores)
					confidence = scores[classId]					
					if confidence > self.confidence_treshold:
						center_x = int(detection[0] * box_scale_w)
						center_y = int(detection[1] * box_scale_h)
						width = int(detection[2] * box_scale_w)
						height = int(detection[3] * box_scale_h)
						left = int(center_x - width / 2)
						top = int(center_y - height / 2)
						classIds.append(classId)
						confidences.append(float(confidence))
						boxes.append([left, top, width, height])	

			indices = cv.dnn.NMSBoxes(boxes, confidences, self.confidence_treshold, self.confidence_treshold-0.1)

			if len(indices) > 0:
				for i in indices.flatten():
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])
					color = self.RECTANGLE_COLOR
					cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
					text = "{}: {:.4f}".format(self.classes[classIds[i]], confidences[i])
					cv.putText(frame, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
		
			cv.imshow('Human Detection',frame)			
			cv.waitKey(1)

def main(args):
	rospy.init_node('human_detection', anonymous=True)

	hd_ = HumanDetetcion()

	processFramesThread = Thread(target=hd_.processingThread)
	#processFramesThread.start()

	postProcessFramesThread = Thread(target=hd_.postProcessThread)
	#postProcessFramesThread.start()
	
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
		cv.destroyAllWindows()
		hd_.process = False
		#processFramesThread.join()
		#postProcessFramesThread.join()

if __name__ == '__main__':
    main(sys.argv)
		
