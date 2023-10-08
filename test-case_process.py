import csv
from enum import Enum
import math

class Vehicle(Enum):
    car=1
    truck_bus=2
    bicycle=3
    pedestrian=4

with open("00_recordingMeta.csv", mode="r", encoding="utf-8-sig") as f_recording:
    recording_reader = csv.reader(f_recording)
    recording_header = next(recording_reader)
    for row in recording_reader:
        duration=row[6]
        num_states=int(row[6]*row[2])
        
cur_count=0
while cur_count<num_states:
    with open("00_tracksMeta.csv", mode="r", encoding="utf-8-sig") as f_tracks:
        tracks_reader = csv.reader(f_tracks)
        tracks_header = next(tracks_reader)
        for row in tracks_reader:
            trackID=row[1]
            initialFrame=row[2]
            finalFrame=row[3]
            numFrames=row[4]
            width=row[5]
            length=row[6]
            vehicleClass=row[7]
            if numFrames>1500: continue # 60s*25Hz=1500frames/s
            if initialFrame>=cur_count: # when the test case starts, the vehicle is not in the map yet.
                
        
        
    
        