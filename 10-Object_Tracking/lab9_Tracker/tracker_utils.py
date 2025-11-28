import cv2
import numpy as np

class MultiObjectTracker:
    def __init__(self, tracker_type="CSRT", use_legacy=True):
        self.tracker_type = tracker_type.upper()
        self.use_legacy = use_legacy and hasattr(cv2, "legacy")
        self.trackers = None
        self.factory = self._get_factory(self.tracker_type)

    def _get_factory(self, tracker_type):
        if self.use_legacy:
            FACTORIES = {
                "KCF":   cv2.legacy.TrackerKCF_create,
                "MIL":   cv2.legacy.TrackerMIL_create,
            }
        else:
            FACTORIES = {
                "KCF":   getattr(cv2, "TrackerKCF_create", None),
                "MIL":   getattr(cv2, "TrackerMIL_create", None),
            }
        factory = FACTORIES.get(tracker_type)
        if factory is None:
            raise ValueError(f"Tracker type '{tracker_type}' is not available in this OpenCV build.")
        return factory

    def _create_multitracker(self):
        if self.use_legacy:
            return cv2.legacy.MultiTracker_create()
        return cv2.MultiTracker_create()

    def reset(self):
        self.trackers = None

    def init_from_boxes(self, frame_or_roi, boxes):
        self.trackers = self._create_multitracker()
        for box in boxes:
            tr = self.factory()
            self.trackers.add(tr, frame_or_roi, tuple(map(float, box)))

    def add(self, frame_or_roi, box):
        if self.trackers is None:
            self.trackers = self._create_multitracker()
        tr = self.factory()
        self.trackers.add(tr, frame_or_roi, tuple(map(float, box)))

    def update(self, frame_or_roi):
        if self.trackers is None:
            return False, []
        ok, boxes = self.trackers.update(frame_or_roi)
        return ok, boxes

    @staticmethod
    def nms(boxes, iou_thr=0.3):
        if not boxes:
            return []
        b = np.array(boxes, dtype=np.float32)
        x1, y1 = b[:, 0], b[:, 1]
        x2, y2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = areas.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            iw = np.maximum(0.0, xx2 - xx1)
            ih = np.maximum(0.0, yy2 - yy1)
            inter = iw * ih
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou < iou_thr]
        return [tuple(map(int, b[i])) for i in keep]