"""
 * @author Ricardo Merlos Torres
 * @email [contact@ricardomerlostorres.com]
 * @create date 2024-11-28 12:20:28
 * @modify date 2024-11-28 14:21:05
 * @desc [description]
"""

import csv
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd


def calculate_times(csv_file_name):
    with open(csv_file_name, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        rows = list(csv_reader) 
        if not rows:
            print("The CSV file is empty.")
            return

        start_time = datetime.strptime(rows[0]["Time"], "%H:%M:%S")
        end_time = datetime.strptime(rows[-1]["Time"], "%H:%M:%S")
        total_time = (end_time - start_time).total_seconds()
        
        print(f"Total Time: {total_time} seconds")

        process_times = {}
        for pad_index in [0, 1]:  
            pad_column = f"pad_{pad_index}_state"
            previous_state = None
            state_start_time = None
            state_durations = {}
            
            for row in rows:
                current_time = datetime.strptime(row["Time"], "%H:%M:%S")
                current_state = row[pad_column]
                
                if current_state != previous_state:
                    if previous_state is not None and previous_state in ["unsoldered", "presoldered", "soldered"]:
                        duration = (current_time - state_start_time).total_seconds()
                        state_durations[previous_state] = state_durations.get(previous_state, 0) + duration
                    
                    state_start_time = current_time
                    previous_state = current_state
            
            if previous_state is not None and previous_state in ["unsoldered", "presoldered", "soldered"]:
                duration = (end_time - state_start_time).total_seconds()
                state_durations[previous_state] = state_durations.get(previous_state, 0) + duration
            
            process_times[f"pad_{pad_index}"] = state_durations

        print("\nProcess Times for Each Pad:")
        for pad, times in process_times.items():
            print(f"{pad}:")
            for state, duration in times.items():
                print(f"  {state}: {duration} seconds")


def plot_cable_trajectories(csv_file_name):
    cable_0_x, cable_0_y = [], []
    cable_1_x, cable_1_y = [], []

    with open(csv_file_name, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            #Cable 0 
            if row["cabel_0_x"] and row["cabel_0_y"] and row["cabel_0_w"] and row["cabel_0_h"]:
                x1 = float(row["cabel_0_x"])
                y1 = float(row["cabel_0_y"])
                width = float(row["cabel_0_w"])
                height = float(row["cabel_0_h"])
                if x1 != 0 and y1 != 0:
                    center_x = x1 + width / 2
                    center_y = y1 - height / 2
                    cable_0_x.append(center_x)
                    cable_0_y.append(center_y)
                else:
                    cable_0_x.append(None)
                    cable_0_y.append(None)

            #Cable 1
            if row["cabel_1_x"] and row["cabel_1_y"] and row["cabel_1_w"] and row["cabel_1_h"]:
                x1 = float(row["cabel_1_x"])
                y1 = float(row["cabel_1_y"])
                width = float(row["cabel_1_w"])
                height = float(row["cabel_1_h"])
                if x1 != 0 and y1 != 0:  
                    center_x = x1 + width / 2
                    center_y = y1 - height / 2
                    cable_1_x.append(center_x)
                    cable_1_y.append(center_y)
                else:
                    cable_1_x.append(None)
                    cable_1_y.append(None)

    plt.figure(figsize=(10, 6))

    if cable_0_x and cable_0_y:
        plt.plot(cable_0_x, cable_0_y, marker='o', label='Cable 0 Trajectory', linestyle='-')

    if cable_1_x and cable_1_y:
        plt.plot(cable_1_x, cable_1_y, marker='o', label='Cable 1 Trajectory', linestyle='-')

    plt.xlim(0, 2448)
    plt.ylim(0, 2048)
    plt.title("Cable Trajectories")
    plt.xlabel("X Coordinate (pixels)")
    plt.ylabel("Y Coordinate (pixels)")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)


def plot_iron_trajectories(csv_file_name):
    iron_0_x, iron_0_y = [], []

    with open(csv_file_name, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            #Cable 0 
            if row["iron_0_x"] and row["iron_0_y"] and row["iron_0_w"] and row["iron_0_h"]:
                x1 = float(row["iron_0_x"])
                y1 = float(row["iron_0_y"])
                width = float(row["iron_0_w"])
                height = float(row["iron_0_h"])
                if x1 != 0 and y1 != 0:
                    center_x = x1 + width / 2
                    center_y = y1 - height / 2
                    iron_0_x.append(center_x)
                    iron_0_y.append(center_y)
                else:
                    iron_0_x.append(None)
                    iron_0_y.append(None)


    plt.figure(figsize=(10, 6))

    if iron_0_x and iron_0_y:
        plt.plot(iron_0_x, iron_0_y, marker='o', label='Iron Trajectory', linestyle='-')

    plt.xlim(0, 2448)
    plt.ylim(0, 2048)
    plt.title("Iron Trajectories")
    plt.xlabel("X Coordinate (pixels)")
    plt.ylabel("Y Coordinate (pixels)")
    plt.legend()
    plt.grid(True)
    plt.show(block=True)

def plot_iron_trajectories_animation(csv_file_name, output_video_name):
    iron_0_x, iron_0_y = [], []

    with open(csv_file_name, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Iron 0
            if row["iron_0_x"] and row["iron_0_y"] and row["iron_0_w"] and row["iron_0_h"]:
                x1 = float(row["iron_0_x"])
                y1 = float(row["iron_0_y"])
                width = float(row["iron_0_w"])
                height = float(row["iron_0_h"])
                if x1 != 0 and y1 != 0:
                    center_x = x1 + width / 2
                    center_y = y1 - height / 2
                    iron_0_x.append(center_x)
                    iron_0_y.append(center_y)
                else:
                    iron_0_x.append(None)
                    iron_0_y.append(None)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 2448)
    ax.set_ylim(0, 2048)
    ax.set_title("Iron Trajectories Animation")
    ax.set_xlabel("X Coordinate (pixels)")
    ax.set_ylabel("Y Coordinate (pixels)")
    ax.grid(True)

    iron_0_line, = ax.plot([], [], 'o-', label='Iron 0 Trajectory', color='red')
    ax.legend()

    def update(frame_idx):
        if frame_idx < len(iron_0_x):
            iron_0_line.set_data(iron_0_x[:frame_idx], iron_0_y[:frame_idx])
        return iron_0_line,

    total_frames = len(iron_0_x)
    ani = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True)
    ani.save(output_video_name, fps=20, extra_args=['-vcodec', 'libx264'])
    plt.close(fig) 

def plot_iron_trajectory_on_video(csv_file_name, video_file_name):
    iron_0_x, iron_0_y = [], []
    
    with open(csv_file_name, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            if row["iron_0_x"] and row["iron_0_y"] and row["iron_0_w"] and row["iron_0_h"]:
                x1 = float(row["iron_0_x"])
                y1 = float(row["iron_0_y"])
                width = float(row["iron_0_w"])
                height = float(row["iron_0_h"])
                if x1 != 0 and y1 != 0:
                    center_x = x1 + width / 2
                    center_y = y1 - height / 2
                    iron_0_x.append(center_x)
                    iron_0_y.append(center_y)
                else:
                    iron_0_x.append(None)
                    iron_0_y.append(None)
    
    cap = cv2.VideoCapture(video_file_name)
    if not cap.isOpened():
        print(f"Error, unable to open video file {video_file_name}")
        return
    
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, video_width)
    ax.set_ylim(video_height, 0) 
    trajectory_line, = ax.plot([], [], 'o-', label='Iron Trajectory', color='red')
    ax.legend()
    plt.grid(True)
    plt.title("Iron Trajectory on Video")
    plt.xlabel("X Coordinate (pixels)")
    plt.ylabel("Y Coordinate (pixels)")
    
    def update(frame_idx):
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            ani.event_source.stop()
            return
        
        ax.imshow(frame, extent=[0, video_width, video_height, 0])
        
        trajectory_line.set_data(iron_0_x[:frame_idx], iron_0_y[:frame_idx])

    ani = FuncAnimation(fig, update, frames=total_frames, interval=200000 / fps, repeat=False)
    plt.show()
    
    cap.release()

video_filename = r"C:\Users\ricar\Documents\Programming\makerspace_20241126\videos\soldering_recordings\Stefan\rec_1.avi"
csv_file_name = "./process_logic/output.csv"  
calculate_times(csv_file_name)
#plot_cable_trajectories(csv_file_name)
#plot_iron_trajectories(csv_file_name)
#plot_iron_trajectory_on_video(csv_file_name, video_filename)
plot_iron_trajectories_animation(csv_file_name, 'iron_trajectories.mp4')
