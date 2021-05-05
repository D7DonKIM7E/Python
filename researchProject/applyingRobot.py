import cv2
import pyaudio
import wave
import threading
import time
import subprocess 
import os, sys 
from facenet_pytorch import MTCNN
from utils.image_utils import *
from numpy_ringbuffer import RingBuffer
from matplotlib import cm
import argparse
import json
import torch
import torch.nn as nn
import torch.optim
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from models import *
from best_model import *
from best_model_js import *
from utils.audio_utils import *
from selenium import webdriver
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()  
parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
parser.add_argument('--model', default='./best_model_best/', type=str,
                        help='select model dir')
parser.add_argument('--pop_cat', default=None, type=str,
                        help='pop-up category news, map, lunch, toilet')
parser.add_argument('--freeze', default=None, type=int,
                        help='freeze time of label') 

args = parser.parse_args()

#label['dtype']['label']
with open('label.json','r') as json_file:
	label = json.load(json_file) 
 

#chk device 
if args.gpu is not None:
	print("Use GPU: {} for demo".format(args.gpu))
	torch.cuda.set_device(args.gpu)
else :
	print('Use CPU')

model_dict = {
	"image":
			{
				"gender" : efficientnet.efficientnet_b0(num_classes=2),
				"emotion" : efficientnet.efficientnet_b0(num_classes=5),
				"gaze" : efficientnet.efficientnet_b0(num_classes=5)
			},
	"audio":
			{
				"gender" : aucousticgender.AucousticGender(),
				"tone" : aucoustictone.tone(num_classes=5),
				"intent" : aucoustictone.tone(num_classes=6)
			}}
 
for best in os.listdir(args.model): 
	if best[0] == 'a':
		model_dict['audio'][best[1:best.find('_')]].load_state_dict(torch.load(args.model + best,map_location=args.gpu if args.gpu else 'cpu')['state_dict'])
	else:
		model_dict['image'][best[1:best.find('_')]].load_state_dict(torch.load(args.model + best,map_location=args.gpu if args.gpu else 'cpu')['state_dict'])

args.label = label
args.model = model_dict
 
class DemoRecorder():
	

	# Audio class based on pyAudio and Wave
	def __init__(self,data):
		
		self.open = True
		self.data = data
		self.device = self.data.gpu 
		self.rate = 48000
		self.frames_per_buffer = int(48000 / 10 )
		self.channels = 1
		self.format = pyaudio.paInt16
		self.audio_filename = "temp_audio.wav"
		self.audio = pyaudio.PyAudio()
		self.mtcnn = MTCNN(device=self.device)

		def callback(in_data, frame_count, time_info, flag):
			audio_data = np.frombuffer(in_data,dtype=np.int16) 
			self.ringBuffer.extend(audio_data)
			return None, pyaudio.paContinue

		self.callback = callback        
				
		self.stream = self.audio.open(format=self.format,
								channels=self.channels,
								rate=self.rate,
								input=True,
								input_device_index=25,
								frames_per_buffer = self.frames_per_buffer,
								stream_callback=self.callback
								) 

		self.ringBuffer = RingBuffer(48000 * 3)
		self.ringBuffer.extend(np.zeros(48000 * 3)) 
 
		self.model = data.model
		self.label = data.label 

	# Audio starts being recorded
	def record(self):
		cap = cv2.VideoCapture(0)
		self.stream.start_stream()
		time.sleep(1) 

		#check pop
		pop = None


		while(self.open == True):
			ret, frame = cap.read()
			frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_AREA)
			try:
				boxes, probs = self.mtcnn.detect(frame, landmarks=False)
				frame = draw_bbox(frame, boxes, probs)
				if len(boxes) > 0:
					rois = detect_rois(boxes)
					for roi in rois:
						(start_Y, end_Y, start_X, end_X) = roi
						self.face = frame[start_Y:end_Y, start_X:end_X]  
					igender, gaze, emotion = self.image_inference()  
					# cv2.putText(frame, self.label['gender'][igender], (end_X-50, start_Y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
					# cv2.putText(frame, self.label['gaze'][gaze], (end_X-50, start_Y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
					cv2.putText(frame, self.label['emotion'][emotion], (end_X-50, start_Y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
 
			except Exception as e:
				# print(e) 
				pass  
			
			
			agender, tone, intent = self.audio_inference() 
			# cv2.putText(frame, self.label['gender'][agender], (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
			cv2.putText(frame, self.label['tone'][tone], (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
			cv2.putText(frame, self.label['intent'][intent], (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
			
			
			cv2.imshow('Demo',frame) 
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			

			if self.data.pop_cat and pop is None:
				#news, map, lunch, home
				if self.data.pop_cat == 'news' and \
				(self.label['intent'][intent] == "ask_others" or self.label['intent'][intent] == "question_others"): 
					driver = webdriver.Chrome() 
					story = 'http://www.wikileaks-kr.org/news/articleView.html?idxno=100475'
					driver.get(story)   
					driver.quit() 
					pop = True
				elif self.data.pop_cat == 'map' and \
				(self.label['intent'][intent] == "ask_others" or self.label['intent'][intent] == "question_others"):							
					# rgb 이미지 불러오기
					rgb_image = cv2.imread('data/map.jpeg')
					rgb_image = cv2.resize(rgb_image, (640,480), interpolation=cv2.INTER_AREA)

					# rgb 이미지 보기
					cv2.imshow('rgb_image', rgb_image)
					time.sleep(5)
					pop = True

				elif self.data.pop_cat == 'lunch' and self.label['tone'][tone] ==  "happy" and \
				(self.label['intent'][intent] == "ask_others" or self.label['intent'][intent] == "question_others"):						
					# rgb 이미지 불러오기
					rgb_image = cv2.imread('data/lunch.png')
					# rgb 이미지 보기
					cv2.imshow('rgb_image', rgb_image)
					pop = True

				elif self.data.pop_cat == 'toilet' and \
				(self.label['intent'][intent] == "ask_location" or self.label['intent'][intent] == "question_location"):							
					# rgb 이미지 불러오기
					rgb_image = cv2.imread('data/toilet.gpeg')
					# rgb 이미지 보기
					rgb_image = cv2.resize(rgb_image, (640,480), interpolation=cv2.INTER_AREA)

					cv2.imshow('rgb_image', rgb_image)
					pop = True

				else: 
					pass

			# if self.open==False:
			# 	break
		
		cap.release()
		cv2.destroyAllWindows()

	def audio_inference(self):
		if(not self.ringBuffer.is_full):
			return
		mel_spectrogram = getMELspectrogram(np.array(self.ringBuffer).astype('float32'), self.rate)
		
		mel_spectrogram-=mel_spectrogram.min()
		maximum = mel_spectrogram.max()
		if maximum == 0:
			maximum = 1e-15
		mel_spectrogram/= maximum
		im = np.uint8(cm.gist_earth(mel_spectrogram)*255)[:,:,:3]
		imagesTensor = transforms.Compose([
			transforms.ToTensor(),
			#transforms.Normalize(0.5, 0.5),
			transforms.Grayscale(num_output_channels=1),
		])(im).view(1,1,im.shape[0],im.shape[1]).to(self.device)

		#chk
		gender = self.predict(self.model['audio']['gender'].to(self.device),imagesTensor)
		tone = self.predict(self.model['audio']['tone'].to(self.device),imagesTensor)
		intent = self.predict(self.model['audio']['intent'].to(self.device),imagesTensor)
			
		return gender, tone, intent
        
	def image_inference(self):
		face = cv2.cvtColor(self.face, cv2.COLOR_BGR2GRAY)
		face = cv2.resize(face, (224,224))

		img_tensor = torch.tensor(face/255.0, dtype=torch.float32)
		img_tensor = torch.unsqueeze(img_tensor, 0)
		img_tensor = torch.unsqueeze(img_tensor, 0).to(self.device)

		gender = self.predict(self.model['image']['gender'].to(self.device), img_tensor)
		gaze = self.predict(self.model['image']['gaze'].to(self.device), img_tensor)
		emotion = self.predict(self.model['image']['emotion'].to(self.device), img_tensor)

		return gender, gaze, emotion

 
	def predict(self,model, X):
		with torch.no_grad():
			model.eval()
			outputs = model(X)
		return int(torch.argmax(outputs,dim=1)[0])

	# Finishes the audio recording therefore the thread too   
	def stop(self):
		
		if self.open==True:
			self.open = False
			self.stream.stop_stream()
			self.stream.close()
			self.audio.terminate()
			
		
		pass

	# Launches the audio recording function using a thread
	def start(self):
		self.record()
 
if __name__== "__main__":

	record = DemoRecorder(args)
	record.start() 
	record.stop()
	print ("Done")

