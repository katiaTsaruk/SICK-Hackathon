import os
import pandas as pd
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List
import threading
from collections import Counter
from feature_computation import re_training_management
from feature_computation.bounding_box import BoundingBox

class TrackedObject:
    # Sample structure for detected objects with ID and bounding box
    def __init__(self, object_id: int, bounding_box: BoundingBox):
        self.id = object_id
        self.bounding_box = bounding_box  # Bounding box (x, y, width, height)

def draw_object_on_image(img:np.ndarray, object:TrackedObject, color:tuple, label=None):

    x = object.bounding_box.x
    y = object.bounding_box.y
    w = object.bounding_box.w
    h = object.bounding_box.h
    cv2.rectangle(img, (int(x), int(y)),  (int(x+w), int(y+h)), color, 2)
    if label is None:
        label = 'Object'
    cv2.putText(img, f'{label}: {object.id}', (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

class StatefulTrackedObject (TrackedObject):
    def __init__(self, object_id: int, bounding_box: BoundingBox, state: int):
        super().__init__(object_id, bounding_box)
        self.state = state

class Tracker(ABC):
    def __init__(self, class_ids = 0):
        self.lock = threading.Lock()  # For thread safety during async updates
        if type(class_ids) is list:
            self.class_ids = class_ids
        else:
            self.class_ids = [class_ids]

    @abstractmethod
    def update_internal(self, sensor_input):
        pass

    @abstractmethod
    def get_objects_internal(self) -> List[TrackedObject]:
        pass

    #returns image ids of outliers
    #ids are relative to current input, i.e. current_input is 0, previous is -1, and so on...
    def update(self, sensor_input) -> List[int]:
        with self.lock:
            return self.update_internal(sensor_input)

    def get_objects(self) -> List[TrackedObject]:
        with self.lock:
            return self.get_objects_internal()
    
def nms(
    boxes: np.ndarray,
    nms_thr: float = 0.5,
    det_thr: float = 0.5,
    class_agnostic: bool = True,
) -> np.ndarray:
    """Single class NMS implemented in Numpy."""
    
    if not class_agnostic:
        raise NotImplementedError("not yet implemented")

    keep = np.where(boxes[:, 4] > det_thr)[0]
    boxes = boxes[keep, :]

    scores = boxes[:, 4]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    boxes = boxes[keep, :]
    return boxes

class KalmanFilterTracker(Tracker):
    #single object kalman filter. Assumes that the detection matching this filter always measures the same instance 
    # -> object id is always 0, re-appearing object has the same ID
    # multiple class_ids can be passed. if so, each class_id corresponds to a state of the object 
    # the returned objects state will be that of the most often measured state in the <state_window> last measurements
    # dissapearance threshold -1 means the object cannot disappear
    def __init__(self, 
                 class_ids=0, 
                 appearance_threshold=30, 
                 disappearance_threshold=30, 
                 ratio_threshold=0.9, 
                 state_window = 30, 
                 more_than_one_outlier_detection = False, 
                 bbox_distance_outlier_threshold = 1.0):
        super().__init__(class_ids)
       

        self.kalman_filter = self.initialize_kalman_filter()
        self.object_is_present = False

        #for single object tracking, the id is always 0
        self.object_id = 0

        self.appearance_threshold = appearance_threshold  # Time window for appearance
        self.disappearance_threshold = disappearance_threshold  # Time window for disappearance
        self.ratio_threshold = ratio_threshold  # Threshold for valid/non-measurements ratio
        self.state_window = state_window
        self.current_state = None

        self.measurement_history = []  # Stores recent measurements (None or valid)
        self.state_history = []
        #tracks whether the measurement could have been an 
        #outlier according to the object presence at time of measurement
        self.outlierness_history = [] 

        self.tracking_results = {}

        self.bbox_distance_outlier_threshold = bbox_distance_outlier_threshold
        self.more_than_one_outlier_detection = more_than_one_outlier_detection

    def filter_detections(self, detections):
        #return measurement as x1 y1 x2 y2 score, class_id
        if detections is None:
            return None
        #find the best measurements in detections for this class

        # first, filter for confidence
        result_nms = nms(detections, nms_thr=0.5, det_thr=0.5)

        # Filter only boxes belonging to the specified class
        result_nms = result_nms[np.isin(result_nms[:, 5], self.class_ids)]

        if result_nms.shape[0] == 0:
            return None

        if result_nms.shape[0] > 1 and self.more_than_one_outlier_detection:
            print("'more than one' outlier detected")
            re_training_management.instance.report_outlier(
                        str(id(self)), 
                        debug_image = None, 
                        frame_id = None, 
                        relative_frame_id = -1,
                        comment="more than one")
            
        max_score_index = np.argmax(result_nms[:, 4])
        box = result_nms[max_score_index]

        return box[0:6].astype(np.float32)

    def is_current_history_relevant(self):
        #check whether the current history is relevant for the object to become present
        #if not object was not seen for enough frames, the point of appearance does not change if the history up to the current point is set to 0
        required_measurements = int(self.appearance_threshold * self.ratio_threshold)
        allowed_non_measurements = self.appearance_threshold - required_measurements
        #the history is irrelevant, if its end already contains enough non-measurements so that the object cannot become present
        return sum(self.measurement_history[-allowed_non_measurements:]) != 0

    def initialize_kalman_filter(self):
        kf = cv2.KalmanFilter(4, 4)
        kf.transitionMatrix = np.eye(4, dtype=np.float32)
        kf.measurementMatrix = np.eye(4, dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 1e8
        kf.statePost = np.zeros((4, 1), dtype=np.float32)
        return kf
    
    def update_object_state(self, maybe_new_measured_state):
        #update the state history and pop front if necessary
    
        self.state_history.append(maybe_new_measured_state)
 
        if len(self.state_history)>self.state_window:
            self.state_history.pop(0)

        
        list_of_valid_states = [item for item in self.state_history if item is not None]
        if list_of_valid_states:
            self.current_state = Counter(list_of_valid_states).most_common(1)[0][0]#get the most common recorded state

    #popped_measurement_history_entry is None, if there was no entry popped, else 0 or 1 according to measurement presence
    #assume this is called only within update_object_presence 
    def check_for_presence_outliers(self):
        new_measurement_history_entry = self.measurement_history[-1]
        could_new_measurement_be_outlier = self.object_is_present != (new_measurement_history_entry == 1)
        self.outlierness_history.append(could_new_measurement_be_outlier)

        # keep outlier history only as long as needed
        max_history_length = max(self.appearance_threshold, self.disappearance_threshold)
        if len(self.outlierness_history) > max_history_length:
            self.outlierness_history.pop(0)

        #check if there was an appearing outlier
        if self.appearance_threshold >= 0 and len(self.outlierness_history) >= self.appearance_threshold:
            reference_outlier_history_event = self.outlierness_history[-self.appearance_threshold]
            #if there was a possible outlier in the past that stated object presence, check if the object indeed appeared
            if self.measurement_history[-self.appearance_threshold] == 1 and reference_outlier_history_event and not self.object_is_present:
                #print("false positive outlier detected")
                re_training_management.instance.report_outlier(
                        str(id(self)), 
                        debug_image = None, 
                        frame_id = None, 
                        relative_frame_id = -self.appearance_threshold,
                        comment = "false positive")

        #check if there was an disappearing outlier       
        if self.disappearance_threshold >= 0 and len(self.outlierness_history) >= self.disappearance_threshold:
            reference_outlier_history_event = self.outlierness_history[-self.disappearance_threshold]
            #if there was a possible outlier in the past that stated object absence, check if the object indeed disappeared
            if self.measurement_history[-self.disappearance_threshold] == 0 and reference_outlier_history_event and self.object_is_present:
                #print("false negative outlier detected")
                re_training_management.instance.report_outlier(
                        str(id(self)), 
                        debug_image = None, 
                        frame_id = None, 
                        relative_frame_id = -self.disappearance_threshold,
                        comment = "false negative")
    

    def check_for_distance_outlier(self, single_measurement):
        # Get the current state (mu) and covariance (S)
        current_state = self.kalman_filter.statePre  # This is the estimated state (mu)
        current_covariance = self.kalman_filter.errorCovPre  # This is the covariance (S)
        sensor_input = single_measurement.reshape(4,1)

        # Calculate the difference (x - mu)
        diff = sensor_input - current_state

        # Inverse of the covariance matrix
        covariance_inv = np.linalg.inv(current_covariance)

        # Mahalanobis distance
        mahalanobis_distance = np.sqrt(diff.T @ covariance_inv @ diff)
        mahalanobis_distance = float(mahalanobis_distance)

        # Create a DataFrame for the new value
        
        
        #if not os.path.exists("./output/mahalanobis_dist.csv"):
        #    pd.DataFrame(columns=['mahalanobis_distance']).to_csv("./output/mahalanobis_dist.csv", index=False)
        #new_data = pd.DataFrame({'Values': [mahalanobis_distance]})
        # Append the new data to the existing CSV file
        # new_data.to_csv("./output/mahalanobis_dist.csv", mode='a', header=False, index=False)

        if mahalanobis_distance > self.bbox_distance_outlier_threshold:
              #print("bbox distance outlier detected")
              re_training_management.instance.report_outlier(
                        str(id(self)), 
                        debug_image = None, 
                        frame_id = None, 
                        relative_frame_id = -1,
                        comment = "bbox_distance exceeded")


    def update_object_presence(self, measured_object_presence:bool):
        # Update measurement history (1 for valid, 0 for false)
        measured_object_presence_int = 1 if measured_object_presence else 0
        self.measurement_history.append(measured_object_presence_int)

        # Keep the history within the appearance/disappearance threshold window
        max_history_length = max(self.appearance_threshold, self.disappearance_threshold)
        if len(self.measurement_history) > max_history_length+1:
           self.measurement_history.pop(0)

        # Calculate the valid and non-valid measurement ratios
        valid_measurements_ratio = sum(self.measurement_history[-self.appearance_threshold:]) / self.appearance_threshold
        non_measurements_ratio = 0.0
        if self.disappearance_threshold >= 0:
            non_measurements_ratio = (self.disappearance_threshold - sum(self.measurement_history[-self.disappearance_threshold:])) / self.disappearance_threshold

        # Handle object appearance
        if not self.object_is_present:
            if valid_measurements_ratio > self.ratio_threshold:
                self.object_is_present = True
        else:
            # Handle object disappearance
            if non_measurements_ratio > self.ratio_threshold:
                self.object_is_present = False
         
        if re_training_management.instance is not None:
            self.check_for_presence_outliers()

    #overwrite for Tracker class
    def update_internal(self, detections):
        measurement = self.filter_detections(detections)

        if measurement is None:
            self.update_object_state(None)
            self.update_object_presence(False)
        else:
            self.update_object_state(measurement[5])
            self.update_object_presence(True)

        
        #in any case proceed the state by 1 time step
        self.kalman_filter.predict()
        self.tracking_results = self.kalman_filter.statePre

        #if there's a measurement, correct with it
        if not measurement is None:
            if re_training_management.instance is not None:
                self.check_for_distance_outlier(measurement[:4])
            self.kalman_filter.correct(measurement[:4])
            self.tracking_results = self.kalman_filter.statePost

    def get_objects_internal(self) -> List[TrackedObject]:
        ret = []
        if self.object_is_present:
            x1 = self.tracking_results[0]
            y1 = self.tracking_results[1]
            x2 = self.tracking_results[2]
            y2 = self.tracking_results[3]
            # kalman filter tracks x1, x2, y1, y2, but TrackedObject has Bounding Box as x,y,w,h

            ret.append(
                StatefulTrackedObject(
                    self.object_id, 
                    BoundingBox(x1, y1, x2-x1, y2-y1),
                    self.current_state))
       
        return ret

class MultiObjectTracker(Tracker):
    def __init__(self, class_ids=0, 
                 appearance_threshold=30, 
                 disappearance_threshold=30, 
                 ratio_threshold=0.9, 
                 iou_threshold=0.3, 
                 state_window = 30):
        
        super().__init__(class_ids)

        self.appearance_threshold = appearance_threshold
        self.disappearance_threshold = disappearance_threshold
        self.ratio_threshold = ratio_threshold
        self.iou_threshold = iou_threshold  # Threshold for IoU-based association
        self.state_window = state_window

        self.trackers: dict[int, KalmanFilterTracker] = {}  # Dictionary to store KalmanFilterTracker objects by object_id
        self.object_id_counter = 0  # To assign unique IDs to new objects

    def iou(self, boxA, boxB):
        # Compute the IoU between two bounding boxes
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def vectorized_IOU(self, boxA, boxB):
        if not boxA or not boxB:
            return []
        # Expand dimensions to allow broadcasting
        boxA = np.expand_dims(boxA, axis=1)  # Shape (N, 1, 4)
        boxB = np.expand_dims(boxB, axis=0)  # Shape (1, M, 4)

        # Calculate the coordinates of the intersection boxes
        xA = np.maximum(boxA[:, :, 0], boxB[:, :, 0])
        yA = np.maximum(boxA[:, :, 1], boxB[:, :, 1])
        xB = np.minimum(boxA[:, :, 2], boxB[:, :, 2])
        yB = np.minimum(boxA[:, :, 3], boxB[:, :, 3])

        # Compute the intersection area
        interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)

        # Compute the area of both sets of boxes
        boxAArea = (boxA[:, :, 2] - boxA[:, :, 0] + 1) * (boxA[:, :, 3] - boxA[:, :, 1] + 1)
        boxBArea = (boxB[:, :, 2] - boxB[:, :, 0] + 1) * (boxB[:, :, 3] - boxB[:, :, 1] + 1)

        # Compute the IoU
        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou

    def associate_detections(self, detections):
        associations = {}
        unassociated_detections = []
        
        #print("detections: ", len(detections))
        #print("existing trackers: ", len(self.trackers))

        detection_boxes = [det[:4] for det in detections]
        tracker_boxes = [tracker.tracking_results[:4].flatten() for _, tracker in self.trackers.items()]

        precomputed_IOU = self.vectorized_IOU(detection_boxes, tracker_boxes)

        for detection_i, det in enumerate(detections):
            best_iou = 0
            best_id = None

            for tracker_i, (obj_id, _) in enumerate(self.trackers.items()):
                iou_score = precomputed_IOU[detection_i, tracker_i]

                if iou_score > best_iou and iou_score > self.iou_threshold:
                    best_iou = iou_score
                    best_id = obj_id

            if best_id is not None:
                associations[best_id] = np.array([det])
            else:
                unassociated_detections.append(det)

        #print("unassociated detections: ", len(unassociated_detections))
        return associations, np.array(unassociated_detections)
    
    #overwrite for Tracker class
    def update_internal(self, sensor_input):
        #remove low confident detections and wrong class
        detections = nms(sensor_input, nms_thr=0.5, det_thr=0.5)
        detections = detections[np.isin(detections[:,5], self.class_ids)]

        # Perform data association based on IoU
        associations, unassociated_detections = self.associate_detections(detections)


        #remember trackers that are present to realize which disappear
        present_trackers = []
        for id, tracker in self.trackers.items():
            if tracker.object_is_present:
                present_trackers.append(id)

        # Update existing trackers with new measurements if they exist
        for obj_id, tracker in self.trackers.items():
            tracker.update(associations.get(obj_id))
        

        # Handle unassociated detections (new objects)
        #print("adding new trackers: ", len(unassociated_detections))
        for measurement in unassociated_detections:
            new_obj_id = self.object_id_counter
            self.object_id_counter += 1

            new_tracker = KalmanFilterTracker(
                class_ids=self.class_ids,
                appearance_threshold=self.appearance_threshold,
                disappearance_threshold=self.disappearance_threshold,
                ratio_threshold=self.ratio_threshold,
                state_window=self.state_window
            )
            new_tracker.update(np.array([measurement]))

            self.trackers[new_obj_id] = new_tracker

        # Remove objects that have just disappeared or cannot appear anymore
        trackers_to_remove = []
        for id, tracker in self.trackers.items():
            if not tracker.object_is_present:
                if not tracker.is_current_history_relevant() or id in present_trackers:
                    trackers_to_remove.append(id)

        #print("removing trackers: ", len(trackers_to_remove))
        for obj_id in trackers_to_remove:
            del self.trackers[obj_id]


    def get_objects_internal(self) -> List[TrackedObject]:
        # Collect tracking results from all active trackers
        tracking_results = []
        for obj_id, tracker in self.trackers.items():
            if tracker.object_is_present:
                tracked_objects = tracker.get_objects_internal()
                for obj in tracked_objects:
                    # Manually set the object_id to match the tracker’s ID
                    obj.id = obj_id
                    tracking_results.append(obj)

        return tracking_results


