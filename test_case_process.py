import csv
from enum import Enum
import math
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES']='2'

class Vehicle(Enum):
    car=1
    truck_bus=2
    bicycle=3
    pedestrian=4
    
    
def process_track(Dtype,n):
    with open("/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_recordingMeta.csv" %{'Dtype':Dtype,'n':n}, mode="r", encoding="utf-8-sig") as f_recording:
        recording_reader = csv.reader(f_recording)
        recording_header = next(recording_reader)
        for row in recording_reader:
            #for highD
            duration=float(row[7])
            num_states=int(duration*float(row[1]))
            #for otherD
            # duration=float(row[6])
            # num_states=int(duration*float(row[2]))
            
            
    #print(num_states)
    cur_count=0
    while cur_count<num_states:
        cur_vehicle_list=[]
        #/DATA1/rzhou/ika/exiD/data/00_recordingMeta.csv
        with open("/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_tracksMeta.csv" %{'Dtype':Dtype,'n':n}, mode="r", encoding="utf-8-sig") as f_tracks:
            tracks_reader = csv.reader(f_tracks)
            tracks_header = next(tracks_reader)
            for row in tracks_reader:
                # for highD
                trackID=row[0]
                initialFrame=float(row[3])
                finalFrame=float(row[4])
                numFrames=float(row[5])
                # for otherD
                # trackID=row[1]
                # initialFrame=float(row[2])
                # finalFrame=float(row[3])
                # numFrames=float(row[4])
                # width=row[5]
                # length=row[6]
                # vehicleClass=row[7]
                if numFrames>1500: continue # 60s*25Hz=1500frames/s
                # when the test case starts, the vehicle is not in the map yet.
                # when the test case ends, the vehicle is already out of the map.
                # as in the cases what we want to learn is the cars get inside from the edge and get outside out the edge of the map
                if initialFrame>=cur_count and finalFrame<=cur_count+1500: 
                    cur_vehicle_list.append(trackID)
            print(cur_vehicle_list)
            
                    
        with open("testcases/%(Dtype)s_%(n)02d_testcases.txt" %{'Dtype':Dtype,'n':n},'a') as f_testcases:
            f_testcases.write(str(cur_count)+"~"+str(cur_count+1500)+": ")
            f_testcases.write(str(cur_vehicle_list))
            f_testcases.write('\n')
            
        cur_count += 1500
                
                
        
type_strs=["exiD","highD","uniD"]

# for n in range(93):        
#     process_track("exiD",n)
    
for n in range(1,61):
    process_track("highD",n)
    
# for n in range(13):
#     process_track("uniD",n)
        