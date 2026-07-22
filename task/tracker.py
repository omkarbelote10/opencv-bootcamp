import cv2

def run_tracker():
    # importing tracker
    tracker = cv2.TrackerCSRT_create()

    # creating video capture obj
    cap = cv2.VideoCapture("first_vid.mp4")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)

    # creating video writer obj
    filename = "first_vid_processed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_size = (frame_width, frame_height)
    video_writer = cv2.VideoWriter(filename=filename,fourcc=fourcc,fps=vid_fps, frameSize=frame_size)

    # selecting object to be tracked
    ret, frame = cap.read()
    w= "bbox selection"
    cv2.namedWindow(w, cv2.WINDOW_NORMAL)
    bbox = cv2.selectROI(w, frame) # bbox = [x, y, w, h]
    cv2.destroyWindow(w)

    # initiate the tracker
    ok = tracker.init(frame, bbox)



    cv2.namedWindow(w, cv2.WINDOW_NORMAL)


    # initiating the loop
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        ok, bbox = tracker.update(frame)
        if not ok:
            cv2.putText(frame, "object not detected", (2, 820), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color = (255, 0, 0), thickness= 2)
        else:
            p1 = (bbox[0], bbox[1])
            p2 = (bbox[0] + bbox[2], bbox[1]+bbox[3])

            p1_int = (int(p1[0]), int(p1[1]))
            p2_int = (int(p2[0]), int(p2[1]))

            cv2.rectangle(frame, p1_int,p2_int,(0, 0, 255), thickness= 2)

        video_writer.write(frame)
        cv2.imshow(w, frame)

        if cv2.waitKey(1) == 27:
            break


    cv2.destroyWindow(w)
    cap.release()
    video_writer.release()
