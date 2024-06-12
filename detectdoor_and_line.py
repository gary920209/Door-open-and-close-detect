'''
Computer Vision @ NTUEE 2024 Fall
Final Project - Door Open and Close  Detection
Author  : Austin, Gary, Ryan, Bowen
Date    : 2024/6/7
'''

import cv2
import numpy as np
import math
import json
import os
import dill
from ultralytics import YOLO

from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
def door_frame_detect(input_videos, output_dir):
    '''
    The function is used to detect the door frame in the video.

    Arg(s):
        input_video_path : Path to the input videos
        output_video_path: Path to the output video (with door frame marked)

    Return(s):
        x1, y1, x2, y2: Coordinates of the door frame
    '''

    # Load the model
    model = YOLO('yolonano.pt')

        
    variables = {}
    for input_video_path in input_videos:
        # Open the video file
        
        cap = cv2.VideoCapture(input_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_name = os.path.basename(input_video_path).split('.')[0]
        output_video_path = os.path.join(output_dir, f'{video_name}_output.mp4')

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        i = 0
        open_x1 = []
        open_y1 = []
        open_x2 = []
        open_y2 = []
        while cap.isOpened():
            i += 1
            ret, frame = cap.read()
            if not ret:
                break
            if i % 2 == 0:
                continue

            results = model(frame)
            for result in results:
                boxes = result.boxes.xyxy
                confs = result.boxes.conf  # Get confidence scores
                classes = result.boxes.cls  # Get class predictions
                print(classes)

                for box, conf, cls in zip(boxes, confs, classes):
                    if cls == 0:
                        x1, y1, x2, y2 = map(int, box)  # Convert to int
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    if cls == 1:
                        x1, y1, x2, y2 = map(int, box)  # Convert to int
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        #save Coordinates for open door and frame
                        open_x1.append(x1)
                        open_y1.append(y1)
                        open_x2.append(x2)
                        open_y2.append(y2)
                    
        # find median of coordinates
        open_x1_med = np.median(open_x1) if open_x1 else 0
        open_y1_med = np.median(open_y1) if open_y1 else 0
        open_x2_med = np.median(open_x2) if open_x2 else 0
        open_y2_med = np.median(open_y2) if open_y2 else 0
        open_x1_med = int(open_x1_med*960/width)
        open_x2_med = int(open_x2_med*960/width)
        open_y1_med = int(open_y1_med*960/height)
        open_y2_med = int(open_y2_med*960/height)

        # Write the frame to the output video
        out.write(frame)
        variables[f"{video_name}_x1"] = open_x1_med
        variables[f"{video_name}_y1"] = open_y1_med
        variables[f"{video_name}_x2"] = open_x2_med
        variables[f"{video_name}_y2"] = open_y2_med
        cap.release()
        out.release()
    return variables

def hough_line_detect(video_path, output_path, X1, Y1, X2, Y2):
    '''
    The function is used to detect the door open and close events in the video.
    '''
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    WIDTH = 960
    HEIGHT = 960
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_path, fourcc, fps, (WIDTH, HEIGHT))
  
    frame_num = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FRAME = 5
    prev_distance = -1
    distances = []
    if abs(Y2 - Y1) < abs(X2 - X1):
        VERTICAL = True
    else:
        VERTICAL = False
        
    RIGHT = False
    LX1 = X1
    LY1 = int((Y1 + Y2) / 2)
    LX2 = X2
    LY2 = int((Y1 + Y2) / 2)
    if VERTICAL:
        LX1 = int((X1 + X2) / 2)
        LY1 = Y1
        LX2 = int((X1 + X2) / 2)
        LY2 = Y2
    if RIGHT:
        LX1 = X2
        LY1 = Y1
        LX2 = X2
        LY2 = Y2
    x1s = []
    x2s = []
    y1s = []
    y2s = []

    epsilon = 5
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break   
        
        # resize the frame to 960x960
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        if frame_num == 0:
            # check if all the corners of frame is black
            # if frame[0, 0][0] == 0 and frame[0, 0][1] == 0 and frame[0, 0][2] == 0 and frame[0, 959][0] == 0 and frame[0, 959][1] == 0 and frame[0, 959][2] == 0 and frame[959, 0][0] == 0 and frame[959, 0][1] == 0 and frame[959, 0][2] == 0 and frame[959, 959][0] == 0 and frame[959, 959][1] == 0 and frame[959, 959][2] == 0: 
            if np.count_nonzero(frame[50:80,WIDTH-80:WIDTH-50,:] < 5) > 2430 and np.count_nonzero(frame[HEIGHT-80:HEIGHT-50,50:80,:] < 5) > 2430:
                fisheye_detect = True    
                if fisheye_detect:
                    if not VERTICAL:
                        if Y2 > 900:
                            Y2 = 800
                        if Y1 < 60:
                            Y1 = 160
                    else:
                        if X2 > 900:
                            X2 = 800
                        if X1 < 60:
                            X1 = 160  
                    LX1 = X1
                    LY1 = int((Y1 + Y2) / 2)
                    LX2 = X2
                    LY2 = int((Y1 + Y2) / 2)
                    if VERTICAL:
                        LX1 = int((X1 + X2) / 2)
                        LY1 = Y1
                        LX2 = int((X1 + X2) / 2)
                        LY2 = Y2
                    if RIGHT:
                        LX1 = X2
                        LY1 = Y1
                        LX2 = X2
                        LY2 = Y2   
                       
                
        frame_num += 1
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get the region of interest (door frame)
        bbox_img = gray[Y1:Y2, X1:X2]
        
        # Apply histogram equalization to the region of interest
        equalized_bbox = cv2.equalizeHist(bbox_img)

        # Replace the region of interest with the equalized image
        gray[Y1:Y2, X1:X2] = equalized_bbox
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edge = cv2.Canny(blurred, 100, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edge, 1, np.pi / 180, threshold=200, minLineLength=min(abs(Y1-Y2), abs(X1-X2))/5, maxLineGap=min(abs(Y1-Y2), abs(X1-X2))/50)
        
        if lines is None:
            continue
        cur_x1s = []
        cur_x2s = []
        cur_y1s = []
        cur_y2s = []
        # Find the line in lines that is in the above rectangle and is closest to the line above
        closest_line = None
        closest_distance = float('inf')
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1!=x2 and y1!=y2:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Check if the line is within the rectangle
                if x1 >= X1 and x1 <= X2 and y1 >= Y1 and y1 <= Y2 and x2 >= X1 and x2 <= X2 and y2 >= Y1 and y2 <= Y2 and 160 > abs(math.atan2(y1 - y2, x1 - x2))*180/math.pi > 20:
                # if x1 >= X1 and x1 <= X2 and y1 >= Y1 and y1 <= Y2 and x2 >= X1 and x2 <= X2 and y2 >= Y1 and y2 <= Y2 and (180 > abs(math.atan2(y1 - y2, x1 - x2))*180/math.pi > 110 or 70 > abs(math.atan2(y1 - y2, x1 - x2))*180/math.pi > 0):
                    cur_x1s.append(x1)
                    cur_x2s.append(x2)
                    cur_y1s.append(y1)
                    cur_y2s.append(y2)
                    # Calculate the distance between the line and the line above
                    distance = np.sqrt((y1 - LY1)**2 + (y2 - LY1)**2)
                    threshold = abs((Y1 - Y2) / 6)
                    if VERTICAL:
                        distance = np.sqrt((x1 - LX1)**2 + (x2 - LX1)**2)
                        threshold = abs((X1 - X2) / 6)
                    if distance < closest_distance and (abs(distance - prev_distance) < threshold or prev_distance == -1):
                        closest_line = line
                        closest_distance = distance

        if len(cur_x1s) > 0:
            x1s.append(np.mean(cur_x1s))
            x2s.append(np.mean(cur_x2s))
            y1s.append(np.mean(cur_y1s))
            y2s.append(np.mean(cur_y2s))
        else:
            x1s.append(x1s[-1] if len(x1s) > 0 else -1)
            x2s.append(x2s[-1] if len(x2s) > 0 else -1)
            y1s.append(y1s[-1] if len(y1s) > 0 else -1)
            y2s.append(y2s[-1] if len(y2s) > 0 else -1)

        # Draw the closest line
        if closest_line is not None:
            x1, y1, x2, y2 = closest_line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            prev_distance = closest_distance
        distances.append(prev_distance)
        if frame_num >= FRAME:
            frame_num = 0

        # draw  rectangle
        cv2.rectangle(frame, (X1, Y1), (X2, Y2), (255, 0, 0), 2)
        cv2.line(frame, (LX1, LY1), (LX2, LY2), (255, 0, 0), 2)
        # Write the frame with marked edges to the output video file
        out.write(frame)

    cap.release()
    out.release()
    # smooth the avg_x1s, avg_x2s, avg_y1s, avg_y2s
    window_size = 10

    #plot the avg with x axis as seconds
    frame_number = np.arange(len(distances))
    time = frame_number / fps

    distances = gaussian_filter1d(distances, sigma=5)
    distances = savgol_filter(distances, window_size, 3)
    distances = ema(distances, alpha=0.3)

    m = np.mean(distances)
    print("mean:", m)
    rec = False
    s, e = 0, 0
    s_valid, e_valid = 0, 0

    first_start_open = 0
    start_open, end_open = 0,0
    start_close, end_close = 0,0

    for i in range(len(distances)):
        if distances[i] > m and not rec:
            rec = True
            if s != 0 and i - e < len(distances) / 20:
                continue
            s = i
        if distances[i] < m and rec:
            rec = False
            e = i
            if e - s > len(distances) / 8:
                s_valid, e_valid = s, e          
                for j in range(s_valid, e_valid):
                    if distances[j] > distances[j + 1]:
                        end_open = j
                        break
                for j in range(s_valid - 1, 0, -1):
                    if distances[j] < distances[j - 1]:
                        if not first_start_open:
                            first_start_open = j
                        start_open = j
                        break
                
                for j in range(e_valid, len(distances)):
                    if j == len(distances) - 1:
                        end_close = j
                        break
                    if distances[j] < distances[j + 1]:
                        end_close = j
                        break
                for j in range(e_valid, 0, -1):
                    if distances[j] > distances[j - 1]:
                        start_close = j
                        break
                
    
    close_state_avg = np.mean(np.append(distances[:first_start_open], distances[end_close:]))
    open_state_avg = np.mean(distances[end_open:start_close])
    
    m = (close_state_avg + open_state_avg) / 2
    rec = False
    s, e = 0, 0
    s_valid, e_valid = 0, 0

    g = np.gradient(distances)
    events = []
    ID = 0
    for i in range(len(distances)):
        if distances[i] > m and not rec:
            rec = True
            if s != 0 and i - e < len(distances) / 15:
                continue
            s = i
        if distances[i] < m and rec:
            rec = False
            e = i
            if e - s > len(distances) / 8:
                s_valid, e_valid = s, e            

                start_open, end_open = 0,0
                for j in range(s_valid, e_valid):
                    if distances[j] > distances[j + 1]:
                        end_open = j
                        break
                for j in range(s_valid - 1, 0, -1):
                    # if distances[j] < distances[j - 1]:
                    if distances[j] < distances[j - 1] or abs(distances[j] - close_state_avg) < epsilon: 
                        start_open = j
                        break                
                print(g[start_open])
                while (g[start_open] < 2.5):
                    start_open += 1
                while (g[start_open] > 3):
                    start_open -= 1

                guess_open = (start_open*7 + end_open) / 8 - 2
                if guess_open - int(guess_open) > 0.5:
                    guess_open = int(guess_open) + 1
                else:
                    guess_open = int(guess_open)
                print("start_open:", start_open)
                print("end_open:", end_open)
                print("guess_open:", guess_open)
                ID += 1
                events.append({
                    "state_id": ID,
                    "description": "Opening",
                    "guessed_frame": guess_open
                })

                start_close, end_close = 0,0
                for j in range(e_valid, len(distances)):
                    if j == len(distances) - 1:
                        end_close = j
                        break
                    if distances[j] < distances[j + 1] or abs(distances[j] - close_state_avg) < epsilon:
                    # if distances[j] < distances[j + 1]:
                        end_close = j
                        break
                for j in range(e_valid, 0, -1):
                    if distances[j] > distances[j - 1]:
                        start_close = j
                        break
                while(g[end_close] > -2.5):
                    end_close -= 1
                while(g[end_close] < -3):
                    end_close += 1
                guess_close = (start_close + end_close*3) / 4
                ID += 1
                events.append({
                    "state_id": ID,
                    "description": "Closing",
                    "guessed_frame": int(guess_close)
                })
    return events

def ema(data, alpha=0.3):
    '''
    Compute the Exponential Moving Average (EMA) of the input data.
    '''
    ema_data = np.zeros_like(data)
    ema_data[0] = data[0]
    for i in range(1, len(data)):
        ema_data[i] = alpha * data[i] + (1 - alpha) * ema_data[i - 1]
    return ema_data

# Detect significant trends (plateaus and U-shapes)
def detect_trends(data, threshold=50):
    '''
    Detect significant trends (plateaus and U-shapes) in the input data.
    '''
    trends = []
    for i in range(10, len(data) - 10):
        if abs(data[i] - data[i - 10]) > threshold or abs(data[i] - data[i + 10]) > threshold:
            trends.append(i)
    return trends


# Find key timesteps where at least two series show significant changes
def find_key_timesteps(*trends):
    '''
    Find key timesteps where at least two series show significant changes.
    input: trends: list of significant trends
    output: key_timesteps: list of key timesteps
    '''
    all_trends = np.concatenate(trends)
    unique, counts = np.unique(all_trends, return_counts=True)
    key_timesteps = unique[counts >= 2]
    return key_timesteps

def process_videos(video_list):
    '''
    The function is used to process the videos and generate the annotations.
    input: video_list: list of video information
    output: video_annotations: list of video annotations
    '''
    video_annotations = []
    for video_info in video_list:
        video_filename = video_info["video_filename"]
        x1, y1, x2, y2  = video_info["bbox"]
        annotations = hough_line_detect("./Tests/"+video_filename,"output.mp4", x1, y1, x2, y2)
        video_annotations.append({
            "video_filename": video_filename,
            "annotations": [
                {
                    "object": "Door",
                    "states": annotations
                }
            ]
        })
    return video_annotations


'''
main function
please fill in the input video path and output video path
'''

input_videos = ['Tests/01.mp4',
                'Tests/02.mp4',
                'Tests/03.mp4',
                'Tests/04.mp4',
                'Tests/05.mp4',
                'Tests/06.mp4',
                'Tests/07.mp4',
                'Tests/08.mp4',
                'Tests/09.mp4',
                'Tests/10.mp4']

variables = door_frame_detect(input_videos, '/output')

# fill in the video filenames and bounding boxes
videos = [
   {"video_filename": "01.mp4", "bbox": (variables["01_x1"], variables["01_y1"], variables["01_x2"], variables["01_y2"])},
   {"video_filename": "02.mp4", "bbox": (variables["02_x1"], variables["02_y1"], variables["02_x2"], variables["02_y2"])},
   {"video_filename": "03.mp4", "bbox": (variables["03_x1"], variables["03_y1"], variables["03_x2"], variables["03_y2"])},
   {"video_filename": "04.mp4", "bbox": (variables["04_x1"], variables["04_y1"], variables["04_x2"], variables["04_y2"])},
   {"video_filename": "05.mp4", "bbox": (variables["05_x1"], variables["05_y1"], variables["05_x2"], variables["05_y2"])},
   {"video_filename": "06.mp4", "bbox": (variables["06_x1"], variables["06_y1"], variables["06_x2"], variables["06_y2"])},
   {"video_filename": "07.mp4", "bbox": (variables["07_x1"], variables["07_y1"], variables["07_x2"], variables["07_y2"])},
   {"video_filename": "08.mp4", "bbox": (variables["08_x1"], variables["08_y1"], variables["08_x2"], variables["08_y2"])},
   {"video_filename": "09.mp4", "bbox": (variables["09_x1"], variables["09_y1"], variables["09_x2"], variables["09_y2"])},
   {"video_filename": "10.mp4", "bbox": (variables["10_x1"], variables["10_y1"], variables["10_x2"], variables["10_y2"])}
]

video_annotations = process_videos(videos)

json_result = {"videos": video_annotations}
with open("output.json", "w") as json_file:
    json.dump(json_result, json_file, indent=4)

