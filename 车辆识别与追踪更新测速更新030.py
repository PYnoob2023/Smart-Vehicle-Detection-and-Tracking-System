import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
from ColorTracker import ColorTracker
import tkinter as tk
from tkinter import filedialog

class VideoProcessor:
    def __init__(self):
        self.model = YOLO('yolov8s.pt')
        self.speed = {}
        self.area = [(225, 335), (803, 335), (962, 408), (962, 408)]
        self.area_c = set()
        self.tracker = ColorTracker()
        self.speed_limit = 62
        self.top_left = self.bottom_right = (0, 0)
        self.drawing = False

    def process_frame(self, frame):
        try:
            if self.drawing:
                cv2.rectangle(frame, self.top_left, self.bottom_right, (0, 255, 0), 2)

            if self.bottom_right != (0, 0):
                cv2.polylines(frame, [np.array(self.area, np.int32)], True, (0, 255, 0), 2)

                results = self.model.predict(frame)
                a = results[0].boxes.data
                px = pd.DataFrame(a.cpu().numpy()).astype("float")

                vehicle_list = []
                for index, row in px.iterrows():
                    x1 = int(row[0])
                    y1 = int(row[1])
                    x2 = int(row[2])
                    y2 = int(row[3])
                    d = int(row[5])
                    c = self.class_list[d]
                    if 'car' in c:
                        vehicle_list.append([x1, y1, x2, y2])

                bbox_id = self.tracker.update(frame, vehicle_list)

                for bbox in bbox_id:
                    x3, y3, x4, y4, id = bbox
                    cx = int(x3 + x4) // 2
                    cy = int(y3 + y4) // 2
                    results = cv2.pointPolygonTest(np.array(self.area, np.int32), ((cx, cy)), False)
                    if results >= 0:
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2,
                                    cv2.LINE_AA)
                        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                        self.area_c.add(id)
                        now = time.time()
                        if id not in self.speed:
                            self.speed[id] = now
                        else:
                            try:
                                prev_time = self.speed[id]
                                self.speed[id] = now
                                px_dist = abs(x4 - x3)
                                real_dist = 2
                                meters_per_pixel = real_dist / px_dist

                                fps = 30
                                time_interval = now - prev_time

                                speed_mps = meters_per_pixel / time_interval
                                speed_kph = speed_mps * 3.6
                                cv2.putText(frame, f"{int(speed_kph)} Km/h", (x4, y4), cv2.FONT_HERSHEY_COMPLEX,
                                            0.8, (0, 255, 255), 2, cv2.LINE_AA)
                                self.speed[id] = now
                            except ZeroDivisionError:
                                pass

                cnt = len(self.area_c)
                cv2.putText(frame, ('Vehicle-Count:-') + str(cnt), (452, 50), cv2.FONT_HERSHEY_TRIPLEX, 1,
                            (102, 0, 255), 2, cv2.LINE_AA)

            return frame
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame

    def draw_rectangle(self, event, x, y, flags, params):
        try:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.top_left = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.bottom_right = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.area = [self.top_left, (self.bottom_right[0], self.top_left[1]), self.bottom_right, (self.top_left[0], self.bottom_right[1])]
                print("Top Left:", self.top_left)
                print("Bottom Right:", self.bottom_right)
                print("Updated area:", self.area)
        except Exception as e:
            print(f"Error in draw_rectangle: {e}")

def main():
    count = 0
    video_processor = VideoProcessor()

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', video_processor.draw_rectangle)

    file_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4")])
    cap = cv2.VideoCapture(file_path)

    with open("coco.txt", "r") as my_file:
        data = my_file.read()
        video_processor.class_list = data.split("\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 3 != 0:
            continue
        frame = cv2.resize(frame, (1020, 500))
        processed_frame = video_processor.process_frame(frame)
        cv2.imshow("Image", processed_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if cv2.getWindowProperty('Image', 0) < 0:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
