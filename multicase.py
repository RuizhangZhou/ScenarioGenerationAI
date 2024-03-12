import numpy as np
import pandas as pd
import os
import csv
import math

Dtype="inD" 
cur_multi_id=1
interval=100 #可调 100=4分钟
num_tracks_each_case=5 #可调 暂设每个case5辆车


with open("/DATA1/rzhou/ika/multi_testcases/%(Dtype)s_multi.csv" %{'Dtype':Dtype}, "w") as f_multi:
        f_multi.truncate(0)
        f_multi.write("caseID")
        for i in range(1,num_tracks_each_case+1):
            f_multi.write(",x%(i)d,y%(i)d" %{'i':i})
        f_multi.write("\n")


for n in range(33):
    print("n:%(n)d\n" %{'n':n})
    with open("/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_recordingMeta.csv" %{'Dtype':Dtype,'n':n}, mode='r') as f_recordingMeta:
        recordingMeta_dict_reader = csv.DictReader(f_recordingMeta)
        #虽然只有一行，但还是要用for来读recordingMeta_dict_reader
        for row in recordingMeta_dict_reader:
            numFramesCase = float(row["duration"]) * float(row["frameRate"])
            numTracks=int(row["numTracks"])
            print(numFramesCase)
            print(numTracks)
            print(int(numFramesCase//100))
            
            
    # with open("/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_tracksMeta.csv" %{'Dtype':Dtype,'n':n}, mode='r') as f_tracksMeta:
    #     tracksMeta_dict_reader = csv.DictReader(f_tracksMeta)
    cur_frame_interval_start=0
    length = int(numFramesCase//interval) #each case in 100 timesteps
    
    
    # 创建一个元素为空列表的列表
    arr = [[] for _ in range(length)]
    tracksMeta_np = pd.read_csv("/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_tracksMeta.csv" %{'Dtype':Dtype,'n':n}).values
    for trackId in range(numTracks):
        numFrames=tracksMeta_np[trackId][4]
        if(numFrames>1500): continue
        initialIdx=math.ceil(tracksMeta_np[trackId][2]/interval)
        finalIdx=math.ceil(tracksMeta_np[trackId][3]/interval)
        for idx in range(initialIdx,finalIdx-1):
            arr[idx].append(trackId)
    print(arr)
        
        
        

    for i in range(length):
        if len(arr[i])<num_tracks_each_case:continue
        arr[i]=arr[i][:num_tracks_each_case]
        for t in range(i*100,(i+1)*100):
            with open("/DATA1/rzhou/ika/multi_testcases/%(Dtype)s_multi.csv" %{'Dtype':Dtype}, "a") as f_multi,\
            open("/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_tracks.csv" %{'Dtype':Dtype,'n':n}, mode='r') as f_tracks:
                tracks_dict_reader = csv.DictReader(f_tracks)
                coordinates=[]
                for row in tracks_dict_reader:
                    if int(row['trackId']) in arr[i] and int(row['frame']) == t:
                        coordinates.append(float(row["xCenter"]))
                        coordinates.append(float(row["yCenter"]))
                    if len(coordinates)>=2*num_tracks_each_case: break
                f_multi.write(str(cur_multi_id))
                for c in coordinates:
                    f_multi.write(",%(c)f" %{'c':c})
                f_multi.write("\n")
        cur_multi_id+=1
                    
                    
# nohup env CUDA_VISIBLE_DEVICES=0,1,2 python multicase.py >> /home/rzhou/Projects/generating-models-for-test-cases/scenariogenerationai/multi_log/multicase_inD.log 2>&1 &
