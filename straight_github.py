import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

def hough_line_detect(video_path, output_path, X1, Y1, X2, Y2,fisheye_detect):
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
    # rectangle for 09_test.mp4: (0, 400), (350, 800)
    # rectangle for 01(test).mp4/03_test.mp4/05_test.mp4: (150, 0), (810, 400) VERT
    # rectangle for 03.mp4/07_test.mp4: (100, 0), (950, 700) HORIZ (True False)
    # rectangle for 02.mp4: (0, 600), (959, 959) RIGHT (False True)
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
                
            # fisheye_detect = True    
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
        
        # 從原始圖像中剪裁出邊界框
        bbox_img = gray[Y1:Y2, X1:X2]
        
        # 對邊界框內的圖像進行直方圖均衡化
        equalized_bbox = cv2.equalizeHist(bbox_img)

        # 將均衡化後的圖像放回原始圖像的相應位置
        gray[Y1:Y2, X1:X2] = equalized_bbox
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # blurred = gray

        edge = cv2.Canny(blurred, 100, 150, apertureSize=3)
        # lines = cv2.HoughLinesP(edge, 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=10)
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

    fig, ax = plt.subplots()
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

                start_open, end_open = 0,0
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
                
                print("start_open:", start_open, "end_open:", end_open)

                start_close, end_close = 0,0
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
                
                print("start_close:",start_close, "end_close:", end_close)
    
    close_state_avg = np.mean(np.append(distances[:first_start_open], distances[end_close:]))
    open_state_avg = np.mean(distances[end_open:start_close])
    print("close state avg:", close_state_avg)
    print("open state avg:", open_state_avg)
    
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
                print("guess_open:", guess_open)
                ID += 1
                events.append({
                    "state_id": ID,
                    "description": "Opening",
                    "guessed_frame": int(guess_open)
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
                print(g[end_close])
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
    ema_data = np.zeros_like(data)
    ema_data[0] = data[0]
    for i in range(1, len(data)):
        ema_data[i] = alpha * data[i] + (1 - alpha) * ema_data[i - 1]
    return ema_data

# Detect significant trends (plateaus and U-shapes)
def detect_trends(data, threshold=50):
    trends = []
    for i in range(10, len(data) - 10):
        if abs(data[i] - data[i - 10]) > threshold or abs(data[i] - data[i + 10]) > threshold:
            trends.append(i)
    return trends


# Find key timesteps where at least two series show significant changes
def find_key_timesteps(*trends):
    all_trends = np.concatenate(trends)
    unique, counts = np.unique(all_trends, return_counts=True)
    key_timesteps = unique[counts >= 2]
    return key_timesteps

def process_videos(video_list):
    video_annotations = []
    for video_info in video_list:
        video_filename = video_info["video_filename"]
        x1, y1, x2, y2,fisheye_detect  = video_info["bbox"]
        annotations = hough_line_detect("./Tests/"+video_filename,"output.mp4", x1, y1, x2, y2, fisheye_detect)
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

# fill in the video filenames and bounding boxes
videos = [
    {"video_filename": "01.mp4", "bbox": (1, 0, 959, 370, False)},
    {"video_filename": "03.mp4", "bbox": (0, 0, 957, 407, False)},
    {"video_filename": "05.mp4", "bbox": (0, 0, 957, 407, False)},
    {"video_filename": "07.mp4", "bbox": (0, 0, 959, 445, False)},
    {"video_filename": "09.mp4", "bbox": (2, 345, 293, 960, True)}
    # {"video_filename": "02.mov", "bbox": (0, 0, 959, 445, False)}

]

video_annotations = process_videos(videos)

json_result = {"videos": video_annotations}
with open("output.json", "w") as json_file:
    json.dump(json_result, json_file, indent=4)

