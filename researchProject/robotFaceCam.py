import time
import cv2
import json
from utils.image_utils import *

import torch
from facenet_pytorch import MTCNN
from models import efficientnet

class FaceCam():
    # Video class based on openCV
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(device=self.device)
        self.open = True
        self.gender_model = def_model('gender', self.device)
        self.gaze_model = def_model('gaze', self.device)
        self.emotion_model = def_model('emotion', self.device)
        self.multimodal_model = def_model('multimodal', self.device)
        

    def rec(self):
        global label

        cap = cv2.VideoCapture(0)
        
        while(self.open==True):
            timer_start = time.time()

            print('start camera!')
            ret, frame = cap.read()

            try:
                # detect face box and probability
                boxes, probs = self.mtcnn.detect(frame, landmarks=False)

                # draw box on frame
                frame = draw_bbox(frame, boxes, probs)

                # perform only when face is detected
                if len(boxes) > 0:
                    # extract the face rois
                    rois = detect_rois(boxes)
                    for roi in rois:
                        (start_Y, end_Y, start_X, end_X) = roi
                        face = frame[start_Y:end_Y, start_X:end_X]
                        print('detect time: ', time.time()-timer_start)
                    
                    predict_start = time.time()
                    gender_i = predict(self.gender_model, face, self.device)
                    gaze_i = predict(self.gaze_model, face, self.device)
                    emotion_i = predict(self.emotion_model, face, self.device)
                    multimodal_i = predict(self.multimodal_model, face, self.device)

                    cv2.putText(frame, label['gender'][gender_i], (end_X-50, start_Y-55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
                    cv2.putText(frame, label['gaze'][gaze_i], (end_X-50, start_Y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
                    cv2.putText(frame, label['emotion'][emotion_i], (end_X-50, start_Y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
                    cv2.putText(frame, label['multimodal'][multimodal_i], (end_X-50), start_Y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)
                    print('predict time: ', time.time()-predict_start)
            except Exception as e:
                print(e)
                pass
            
            # show the frame
            cv2.imshow('Demo', frame)
            
            # q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Interrupted by user!')
                break

        # clear program and close windows
        cap.release()
        cv2.destroyAllWindows()
        print('All done!')

with open('label.json', 'r') as json_file:
    label = json.load(json_file)

f = FaceCam()
f.rec()
