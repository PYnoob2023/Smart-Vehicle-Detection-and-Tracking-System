import numpy as np
import cv2


class ColorTracker:
    def __init__(self):
        # Store the color histograms of the objects
        self.color_histograms = {}
        # Keep the count of the IDs
        self.id_count = 0

    def update(self, frame, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Convert frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get color histogram of new object
        for rect in objects_rect:
            x, y, w, h = rect
            roi = hsv_frame[y:y + h, x:x + w]
            hist = cv2.calcHist([roi], [0, 1], None, [16, 16], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # Find out if that object was detected already
            same_object_detected = False
            for id, ref_hist in self.color_histograms.items():
                similarity = cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL)

                if similarity > 0.8:
                    self.color_histograms[id] = hist
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.color_histograms[self.id_count] = hist
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean up unused IDs
        self.color_histograms = {k: v for k, v in self.color_histograms.items() if
                                 k in [obj[4] for obj in objects_bbs_ids]}

        return objects_bbs_ids


# Example usage
tracker = ColorTracker()
cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection (replace with your object detection algorithm)
    objects_rect = [[100, 100, 50, 50], [200, 200, 60, 60]]

    # Update tracker
    tracked_objects = tracker.update(frame, objects_rect)

    # Draw tracked objects
    for obj in tracked_objects:
        x, y, w, h, _ = obj
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()