"""
 * @author Ricardo Merlos Torres
 * @email [contact@ricardomerlostorres.com]
 * @create date 2024-11-26 11:43:30
 * @modify date 2024-11-27 11:47:22
 * @desc [description]
"""
import json
import albumentations as A
import cv2
import os

json_path = './dtag_rectangle_20241126/dtag_export/annotations/instances_train.json'
imgs_path = './dtag_rectangle_20241126/dtag_export/train/'
with open(json_path, 'r') as file:
    imgs_json_data = json.load(file)

original_num_imgs_folder = len(imgs_json_data["images"])
last_id_num = imgs_json_data["images"][-1]["id"]
last_id_num_annotations = last_id_num


"""Check different transformations: https://explore.albumentations.ai"""
transformations = {
    "brightness_contrast_blur" : A.Compose([
    A.RandomBrightnessContrast(p=0.7),
    A.Blur(blur_limit=(3,7), p=1.0),
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5)
    ]),
    "grayscale_noise": A.Compose([
        A.ToGray(p=1.0),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.8)
    ]),
    "sepia": A.ToSepia(p=1.0),
    "gamma_correction": A.RandomGamma(gamma_limit=(50, 150), p=1.0),
    "color_jitter": A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
    "bleach": A.RandomSnow(
        snow_point_range=(0.2, 0.4), 
        brightness_coeff=2.0, 
        method="texture", 
        p=1.0
    ),
    "random_shadow": A.Compose([
        A.RandomShadow(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ])
}

def find_times_id_repeated_annotations(annotations_list, id_num):
    counter = 0
    positions = []
    for item in range(len(annotations_list)):
        #print(annotations_list[item])
        if annotations_list[item]["image_id"] == id_num:
            counter += 1
            #print("eureka")
            positions.append(item)
    return positions

for img in range(original_num_imgs_folder):
    file_img_name = imgs_json_data["images"][img]["file_name"]
    full_path = os.path.join(imgs_path, file_img_name)
    image = cv2.imread(full_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    original_image_id = imgs_json_data["images"][img]["id"] 
    
    for transformation_name, transformation in transformations.items():
        transformed_image = transformation(image=image)
        
        
        name, ext = os.path.splitext(file_img_name)
        file_transformed_name = f"{name}_{transformation_name}{ext}"
        transformed_fullfile_path = os.path.join(imgs_path, file_transformed_name)
        

        new_image_id = last_id_num + 1
        imgs_json_data["images"].append({
            "license": "NA",
            "file_name": file_transformed_name,
            "height": 576,  
            "width": 768,   
            "date_captured": imgs_json_data["images"][img]["date_captured"],
            "id": new_image_id
        })
        cv2.imwrite(transformed_fullfile_path, transformed_image["image"])
        
        temp_locations = find_times_id_repeated_annotations(
            imgs_json_data["annotations"], original_image_id
        )
        
        if len(temp_locations) > 0:
            for item in temp_locations:
                temp_dict = imgs_json_data["annotations"][item].copy()
                temp_dict["image_id"] = new_image_id  
                imgs_json_data["annotations"].append(temp_dict)  
        
        last_id_num += 1


with open('./dtag_rectangle_20241126/dtag_export/annotations/output_train.json', 'w') as json_file:
    json.dump(imgs_json_data,json_file ,indent=4)