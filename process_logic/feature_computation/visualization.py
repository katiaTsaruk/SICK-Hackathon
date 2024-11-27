import numpy as np
from typing import Dict
from feature_computation.object_tracking import *
import matplotlib as plt

class ObjectTrackerVisualization:
    def __init__(self, trackers:Dict[str,TrackedObject]):
        self.trackers = trackers

        no_classes = 0
        for (_,t) in trackers.items():
            no_classes += len(t.class_ids)

        colors = plt.cm.get_cmap('hsv', no_classes+1)
        class_colors_flat = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in [colors(i) for i in range(no_classes)]]

        self.class_colors = {}
        i = 0
        for (n,t) in trackers.items():
            state_colors = {}
            for j in t.class_ids:
                state_colors[j] = class_colors_flat[i]
                i += 1
            self.class_colors[n] = state_colors

    def draw_visualization_on_frame(self, img: np.ndarray):
         for  (name, tracker) in self.trackers.items():
                        for o in tracker.get_objects():
                            o_state = tracker.class_ids[0]
                            if isinstance(o, StatefulTrackedObject):
                                o_state = o.state
                            
                            draw_object_on_image(img, o, self.class_colors[name][o_state], name)

class ObjectDetectionVisualization:
    def __init__(self, number_classes:int = 1):
        self.number_classes = number_classes
        colors = plt.cm.get_cmap('hsv', self.number_classes+1)
        self.class_colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in [colors(i) for i in range(self.number_classes)]]
    
    def draw_visualization_on_frame(self, image:np.ndarray, detections_result:np.ndarray, show_scores:bool = False, show_class_id = False, thickness = 2):

        for result in detections_result:
            class_id = result[5]
            [x,y,x2,y2] = result[0:4]
            class_name = "class_" + str(class_id)
            x0 = int(x)
            y0 = int(y)
            x1 = int(x2)
            y1 = int(y2)

            if show_scores and show_class_id:
                text = "{} {:.1f}%".format(class_name, result[4] * 100)
            elif show_class_id and not show_scores:
                text = f"{class_name}"
            elif show_scores and not show_class_id:
                # don't show score for false negatives - they are missed and have no score
                text = "{:.1f}%".format(result[4] * 100)
            else:
                text = ""
            font = cv2.FONT_HERSHEY_SIMPLEX
            padding = 3

            txt_size = cv2.getTextSize(text, font, 0.4, thickness)[0]

            cv2.rectangle(
                image,
                (x0, y0),
                (x1, y1),
                self.class_colors[int(class_id)],
                thickness=thickness,
            )
            cv2.putText(
                image,
                text,
                (x0 + padding, y0 + txt_size[1] + padding),
                font,
                0.4,
                self.class_colors[int(class_id)],
                thickness=1,
            )
         

        