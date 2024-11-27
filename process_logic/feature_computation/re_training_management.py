import pandas as pd
import os
import cv2

class ImageCache:
    def __init__(self, max_history):
        self.max_history = max_history
        self.images = {}
        self.image_counter = 0

    def add_image(self, image):
        self.images[self.image_counter] = image.copy()
        self.image_counter +=1
        #need to keeyp 1 more than max_history
        if self.image_counter > self.max_history + 1:
            del self.images[self.image_counter - self.max_history - 2]
        return self.image_counter - 1

class ReTrainingManagement:
    def __init__(self, max_history, output_dir):
        self.image_cache = ImageCache(max_history)
        self.data = pd.DataFrame(columns=["file_name", "network", "comment"])
        self.output_dir = output_dir

    def report_outlier(self,  network_name, debug_image = None, frame_id = None, relative_frame_id = None, comment:str = ""):
        assert(frame_id is not None or relative_frame_id is not None)
        if frame_id is None:
            frame_id = self.image_cache.image_counter + relative_frame_id
        filepath = self.output_dir + f"/frame_{frame_id:05d}.png"
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_dir + "/debug"), exist_ok=True)
    
        # Check if the file already exists
        if not os.path.isfile(filepath):
            cv2.imwrite(filepath, self.image_cache.images[frame_id])
            print(f"Image saved to {filepath}")

        if debug_image is not None:
            debug_image_index = 0
            while os.path.isfile(self.output_dir + f"/debug/frame_{frame_id:05d}_{debug_image_index:02d}.png"):
                debug_image_index += 1
            cv2.imwrite(self.output_dir + f"/debug/frame_{frame_id:05d}_{debug_image_index:02d}.png", debug_image)

        self.data = pd.concat([self.data,pd.DataFrame({'file_name': [filepath], 'network': [network_name], 'comment':[comment]})], ignore_index=True)
        self.data.to_csv(self.output_dir + "/data.csv")
        


instance:ReTrainingManagement = None
             






        
