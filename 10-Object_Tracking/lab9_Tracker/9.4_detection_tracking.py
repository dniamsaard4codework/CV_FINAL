import cv2
from tracker_utils import MultiObjectTracker

cap = cv2.VideoCapture("video.mp4")
print("Press ESC to exit")

# Initialize the object detector
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
mot = MultiObjectTracker(tracker_type="KCF", use_legacy=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    # print(f"Frame dimensions: {width}x{height}")

    # Extract the region of interest (ROI) for object detection
    roi = frame[height//2:height, 300:1300]

    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    # Create a contour to visualize the detected objects
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    dets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 3)
            dets.append((x, y, w, h))
    
    # auto init when no active trackers and we have detections
    if mot.trackers is None and len(dets) > 0:
        dets_nms = mot.nms(dets, iou_thr=0.3)
        mot.init_from_boxes(roi, dets_nms)

    # update trackers on ROI, draw in green
    success, boxes = mot.update(roi)
    if success and len(boxes) > 0:
        for b in boxes:
            x, y, w2, h2 = map(int, b)
            cv2.rectangle(roi, (x, y), (x + w2, y + h2), (0, 255, 0), 2)
    
    elif mot.trackers is not None and not success:
        # drop trackers if update failed
        mot.reset()
    
    cv2.imshow("Original Frame", frame)
    cv2.imshow("ROI Mask", mask)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:               # ESC key to exit
        break
    elif key == ord('r'):
        mot.reset()
    elif key == ord('s'):
        # manual add on current ROI
        box = cv2.selectROI("Original Frame", roi, fromCenter=False, showCrosshair=True)
        if box is not None and box[2] > 0 and box[3] > 0:
            mot.add(roi, box)

cap.release()
cv2.destroyAllWindows()