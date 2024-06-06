import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

def hough_line_detect(video_path, output_path):
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
    X1 = 0
    Y1 = 0
    X2 = 959
    Y2 = 448
    VERTICAL = True
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # resize the frame to 1280x960
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame_num += 1
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # blurred = gray

        edge = cv2.Canny(blurred, 100, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edge, 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=10)

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
                if x1 >= X1 and x1 <= X2 and y1 >= Y1 and y1 <= Y2 and x2 >= X1 and x2 <= X2 and y2 >= Y1 and y2 <= Y2:
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

    for i in range(len(distances)):
        if distances[i] > m and not rec:
            rec = True
            if s != 0 and i - e < len(distances) / 20:
                continue
            s = i
        if distances[i] < m and rec:
            rec = False
            e = i
            if e - s > e_valid - s_valid:
                s_valid, e_valid = s, e            

    start_open, end_open = 0,0
    for i in range(s_valid, e_valid):
        if distances[i] > distances[i + 1]:
            end_open = i
            break
    for i in range(s_valid - 1, 0, -1):
        if distances[i] < distances[i - 1]:
            start_open = i
            break
    
    print(start_open, end_open)

    start_close, end_close = 0,0
    for i in range(e_valid, len(distances)):
        if distances[i] < distances[i + 1]:
            end_close = i
            break
    for i in range(e_valid, 0, -1):
        if distances[i] > distances[i - 1]:
            start_close = i
            break
    
    print(start_close, end_close)

    # ax.plot(time, x1s, label='avg_x1')
    # ax.plot(time, x2s, label='avg_x2')
    # ax.plot(time, y1s, label='avg_y1')
    # ax.plot(time, y2s, label='avg_y2')
    # ax.plot(time, distances, label='distances')
    ax.plot(distances, label='distances')


    # for t in key_timesteps:
    #     plt.axvline(x=t, color='black', linestyle='--', alpha=0.7)

    ax.legend()
    plt.show()
    # save the line chart to a file
    fig.savefig('line_chart.png')

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






# Replace 'your_video.mp4' with the path to your input video file
# Replace 'output_video.avi' with the desired path for your output video file
# mark_straight_edges('01.mp4', '01_straight_edges_2.mp4')
hough_line_detect('./Tests/03.mp4', './Tests/03_hough_line.mp4')

