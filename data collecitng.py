import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from Commonfunctions import *

# Set up the data path
DATA_PATH = os.path.join('DataSet')

try:
    os.makedirs(DATA_PATH)
except FileExistsError:
    pass


no_sequences = 25
sequence_length = 75

fps = 6  
frame_width = 640
frame_height = 480
fourcc = cv2.VideoWriter_fourcc(*'XVID')  


mp_holistic = mp.solutions.holistic

while True:
    action = input("Enter the meaning of action (type 'stop' to exit): ")
    if action.lower() == "stop":
        break
    elif not action.strip():  
        print("Input was empty. Please enter a valid action.")
        continue

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            for sequence in range(no_sequences):
                print(sequence, "Start collecting data")
                try:
                    
                    os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
                    
                    
                    video_path = os.path.join(DATA_PATH, action, str(sequence), 'sequence.avi')
                    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

                    for frame_num in range(sequence_length):                
                        ret, frame = cap.read()
                        if not ret:
                            print("Failed to capture image")
                            break

                        image, results = mediapipe_detection(frame, holistic)
                        
                        draw_landmarks(image, results)

                        if frame_num < 5:
                            cv2.putText(image, 'READY', (200, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4, cv2.LINE_AA)
                        elif frame_num < 10:
                            cv2.putText(image, '1', (300, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4, cv2.LINE_AA)
                        elif frame_num < 15:
                            cv2.putText(image, '2', (300, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4, cv2.LINE_AA)
                        elif frame_num < 20:
                            cv2.putText(image, '3', (300, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4, cv2.LINE_AA)
                        elif frame_num < 25:
                            cv2.putText(image, 'GO', (300, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4, cv2.LINE_AA)
                        else:
                            cv2.putText(image, 'video sequence {}'.format(sequence), (15, 12), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            
                            keypoints = extract_keypoints(results)
                            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                            img_path =os.path.join(DATA_PATH, action, str(sequence),f'frame_{frame_num}.jpg')
                            np.save(npy_path, keypoints)
                            cv2.imwrite(img_path , frame)
                            
                            
                            out.write(frame)

                        cv2.imshow('OpenCV Feed', image)
                    
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                    print("Data saved for: ", action, "-", sequence)

                    
                    out.release()

                except Exception as e:
                    print("An error occurred:", e)
                    pass

            cap.release()
            cv2.destroyAllWindows()
print("Stopped.")
