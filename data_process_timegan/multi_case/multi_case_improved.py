import numpy as np
import pandas as pd
import os
import csv
import math

Dtype="inD" 
startmap=9
endmap=23
cur_multi_id=1
interval=250 #可调 100=4s 250=10s
num_tracks_each_case=5 #可调 暂设每个case5辆车
num_fea=2*num_tracks_each_case


with open(f"/DATA1/rzhou/ika/multi_testcases/{Dtype}_multi_{startmap:02d}-{endmap}_interval{interval}_numfea{num_fea}_withoutpadding.csv", "w") as f_multi:
        f_multi.truncate(0)
        f_multi.write("caseID")
        for i in range(1,num_tracks_each_case+1):
            f_multi.write(",x%(i)d,y%(i)d" %{'i':i})
        f_multi.write("\n")


for n in range(startmap,endmap+1):
    print("n:%(n)d" %{'n':n})
    with open("/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_recordingMeta.csv" %{'Dtype':Dtype,'n':n}, mode='r') as f_recordingMeta:
        recordingMeta_dict_reader = csv.DictReader(f_recordingMeta)
        #虽然只有一行，但还是要用for来读recordingMeta_dict_reader
        for row in recordingMeta_dict_reader:
            numFramesCase = float(row["duration"]) * float(row["frameRate"])
            numTracks=int(row["numTracks"])
            print(numFramesCase)
            print(numTracks)
            print(int(numFramesCase//interval))
            
            
    # with open("/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_tracksMeta.csv" %{'Dtype':Dtype,'n':n}, mode='r') as f_tracksMeta:
    #     tracksMeta_dict_reader = csv.DictReader(f_tracksMeta)
    cur_frame_interval_start=0
    length = int(numFramesCase//interval) #each case in 100 timesteps
    
    
    # 创建一个元素为空列表的列表
    arr = [[] for _ in range(length)]
    tracksMeta_np = pd.read_csv("/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_tracksMeta.csv" %{'Dtype':Dtype,'n':n}).values
    for trackId in range(numTracks):
        numFrames=tracksMeta_np[trackId][4]
        if(numFrames>5000 or numFrames<interval): continue
        initialIdx=math.ceil(tracksMeta_np[trackId][2]/interval)
        finalIdx=math.ceil(tracksMeta_np[trackId][3]/interval)
        for idx in range(initialIdx,finalIdx-1):
            arr[idx].append(trackId)
    print(arr)
        
        
        
    cur_r_start=0
    tracks_np = pd.read_csv("/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_tracks.csv" %{'Dtype':Dtype,'n':n},low_memory=False).values
    rows_tracks_np = tracks_np.shape[0] 
    with open(f"/DATA1/rzhou/ika/multi_testcases/{Dtype}_multi_{startmap:02d}-{endmap}_interval{interval}_numfea{num_fea}_withoutpadding.csv", "a") as f_multi:
        for i in range(length):
            if len(arr[i])<num_tracks_each_case:continue
            arr[i]=arr[i][:num_tracks_each_case]
            temp_arr = np.ones((num_tracks_each_case, interval,2))#(5,250,2)
            temp_track_id=0
            r=cur_r_start
            while r < rows_tracks_np:
                if tracks_np[r][1] in arr[i] and tracks_np[r][2] == i*interval:
                    if temp_track_id==0: cur_r_start=r+1
                    for a in range(interval):
                        temp_arr[temp_track_id][a][0]=tracks_np[r+a][4]#xCenter
                        temp_arr[temp_track_id][a][1]=tracks_np[r+a][5]#yCenter
                    temp_track_id+=1
                if temp_track_id>=num_tracks_each_case: break
                r+=1
            
            for a in range(interval):
                f_multi.write(str(cur_multi_id))
                for temp_track_id in range(num_tracks_each_case):
                    f_multi.write(",%(c1)f,%(c2)f" %{'c1':temp_arr[temp_track_id][a][0],'c2':temp_arr[temp_track_id][a][1]})
                f_multi.write("\n")
            cur_multi_id+=1
                    
                    
# nohup python multi_case_improved.py >> /home/rzhou/Projects/scenariogenerationai/data_process_timegan/multi_case/multi_log/multicase_inD_18-29.log 2>&1 &
