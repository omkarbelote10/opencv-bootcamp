import cv2
from ultralytics import YOLO
import numpy as np

# drawing rect
def drawRectangle(frame, box):
    x1, y1, x2, y2 = map(int, box)
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    cv2.rectangle(frame,(x1,y1), (x2,y2), (255, 255, 0))
    cv2.circle(frame, (cx, cy), 2, (255, 0, 255), 2)
    
def car_color(frame, box):
        x1, y1, x2, y2 = map(int, box)
        car = frame[y1:y2, x1:x2]
        # red car percentage
        hsv = cv2.cvtColor(car, cv2.COLOR_BGR2HSV)
        lower_red_1 = np.array([0, 70, 50])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 70, 50])
        upper_red_2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        mask = mask1 | mask2

        red_pixels = cv2.countNonZero(mask)
        total_pixels = car.shape[0]*car.shape[1]
        red_percentage = red_pixels / total_pixels

        # white car percentage
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 40, 255])

        mask = cv2.inRange(hsv, lower_white, upper_white)

        white_pixels = cv2.countNonZero(mask)
        total_pixels = car.shape[0]*car.shape[1]
        white_percentage = white_pixels / total_pixels

        cv2.putText(frame,f"red:   {red_percentage:.1f}",(x1, y1),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)
        cv2.putText(frame, f"white:   {white_percentage:.1f}",  (x2, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255, 255, 255), thickness=2)

        if white_percentage > 0.30:
            return "white"
        elif red_percentage > 0.30:
            return "red"
        else:
            return "other"

def run_counter():  
    model = YOLO(r"best.pt")


    # Video capture object
    cap = cv2.VideoCapture("second_vid.mp4")
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


    # Video writer object
    filename = "second_vid_processed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    framesize = (frame_width, frame_height)
    video_writer = cv2.VideoWriter(filename, fourcc = fourcc, fps = fps, frameSize=framesize)

    # initiate some variables
    track_states = {}
    counted = set()
    white_lr_count = 0
    white_rl_count = 0
    red_lr_count = 0
    red_rl_count = 0
    boundry_x = 900
    window_name = "Tracking"
    window = cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # loop start
    while True:
        tick1 = cv2.getTickCount()
        ret, frame = cap.read()
        if not ret:
            break


        results = model.track(frame,
                              persist=True, 
                              tracker="bytetrack.yaml",
                              conf = 0.60,
                              iou = 0.4,
                              classes = [3, 4, 5, 8],
                              verbose=False
                            )

        result = results[0]

        boxes = result.boxes

        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

        for box, track_id in zip(xyxy, ids):
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2)/2)
            cy = int((y1 + y2)/2)

            color = car_color(frame= frame, box= box)

            drawRectangle(frame, box)

            if color == "white":
                previous_state = track_states.get(track_id)
                if cx < boundry_x:
                    current_state = "left"
                elif boundry_x < cx:
                    current_state = "right"

                if current_state == "right" and previous_state == "left":
                    white_lr_count += 1

                if current_state == "left" and previous_state == "right": 
                    white_rl_count += 1

                track_states[track_id] = current_state

            elif color == "red":
                previous_state = track_states.get(track_id)
                if cx < boundry_x:
                    current_state = "left"
                elif boundry_x < cx:
                    current_state = "right"

                if current_state == "right" and previous_state == "left":
                    red_lr_count += 1

                if current_state == "left" and previous_state == "right":
                    red_rl_count += 1

                track_states[track_id] = current_state
            else:
                pass

        cv2.putText(frame, f"white_lr_count: {white_lr_count}", (0, 100), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 1.5, color = (255, 255, 255), thickness=2)
        cv2.putText(frame, f"white_rl_count: {white_rl_count}", (0, 150), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 1.5, color = (255, 255, 255), thickness=2)
        cv2.putText(frame, f"Total Count: {white_rl_count + white_lr_count}", (0, 200), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 1.5, color = (255, 255, 255), thickness=2)
        cv2.putText(frame, f"red_lr_count: {red_lr_count}", (1500, 100), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 1.5, color = (0, 0, 255), thickness=2)
        cv2.putText(frame, f"red_rl_count: {red_rl_count}", (1500, 150), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 1.5, color = (0, 0, 255), thickness=2)
        cv2.putText(frame, f"Total Count: {red_rl_count + red_lr_count}", (1500, 200), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 1.5, color = (0, 0, 255), thickness=2)
        cv2.line(frame, (boundry_x, 0), (boundry_x, frame_height), (255, 0, 0), thickness=1)
    
        tick2 = cv2.getTickCount()
        fre = cv2.getTickFrequency()
    
        fps = 1 / ((tick2 - tick1) / fre)
        cv2.putText(frame, f"fps: {fps:.0f}", (900,150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))

        video_writer.write(frame)
        cv2.imshow(window_name, frame)
    
        key = cv2.waitKey(1)
        if key == 27:
            break
        
    cap.release()
    video_writer.release()
    cv2.destroyWindow(window_name)