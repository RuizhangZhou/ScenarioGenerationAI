import numpy as np
import pandas as pd
import os

# from collections import defaultdict
# def trackId_struct():
#     return defaultdict(frame_struct)
# def frame_struct():
#     return defaultdict(tracks_struct)
# def tracks_struct():
#     return dict(x=0,y=0)

os.environ['CUDA_VISIBLE_DEVICES']='2'

def record_vehicles(Dtype,n):
    
    tracks_np = pd.read_csv("/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_tracks.csv" %{'Dtype':Dtype,'n':n}).values
    
    #clear the text first
    with open("testcases/%(Dtype)s/%(Dtype)s_%(n)02d_testcases_vehicles_test.csv" %{'Dtype':Dtype,'Dtype':Dtype,'n':n}, "w") as f:
        f.truncate(0)
    
    with open("testcases/%(Dtype)s/%(Dtype)s_%(n)02d_testcases_pure.txt" %{'Dtype':Dtype,'Dtype':Dtype,'n':n},'r') as f_testcases,\
    open("testcases/%(Dtype)s/%(Dtype)s_%(n)02d_testcases_vehicles_test.csv" %{'Dtype':Dtype,'Dtype':Dtype,'n':n}, "a") as f_vehicles:
        f_vehicles.write("trackID,frame,x,y\n")
        
        cur_row_f_vehicles=0
        start_frame_of_cur_case=0
        cur_trackId=-1
        for line in f_testcases.readlines():
            line = line.strip('\n')  #去掉列表中每一行的换行符
            veh_id_list=line.split()
            if len(veh_id_list)==0: break
            veh_id_list = list(map(int, veh_id_list))
            first_veh_id_cur_case=veh_id_list[0]
            last_veh_id_cur_case=veh_id_list[-1]
            #veh_3d_dic=defaultdict(trackId_struct)
            veh_3d_dic={}
            trackID_in_case=0
            while cur_row_f_vehicles<len(tracks_np) and tracks_np[cur_row_f_vehicles][1]<=last_veh_id_cur_case:
                #cur_trackId=int(tracks_np[cur_row_f_vehicles][1])
                if int(tracks_np[cur_row_f_vehicles][1]) in veh_id_list:
                    if int(tracks_np[cur_row_f_vehicles][1])!=cur_trackId: 
                        trackID_in_case+=1
                        cur_trackId=int(tracks_np[cur_row_f_vehicles][1])
                    cur_frame=int(tracks_np[cur_row_f_vehicles][2])
                    veh_3d_dic[trackID_in_case,cur_frame-start_frame_of_cur_case]=(tracks_np[cur_row_f_vehicles][4],tracks_np[cur_row_f_vehicles][5])
                    
                    # f_vehicles.write("trackID:%d "%cur_trackId)
                    # f_vehicles.write("frame:%d "%cur_frame)
                    # f_vehicles.write("x:%f "%tracks_np[cur_row_f_vehicles][4])
                    # f_vehicles.write("y:%f "%tracks_np[cur_row_f_vehicles][5])
                cur_row_f_vehicles+=1
            
            for k,v in veh_3d_dic.items():
                f_vehicles.write("%(k0)d,%(k1)d,%(v0)f,%(v1)f\n" %{'k0':k[0],'k1':k[1],'v0':v[0],'v1':v[1]})
                
            # f_vehicles.write(str(veh_3d_dic))
            # f_vehicles.write("\n")
            start_frame_of_cur_case+=1500
                
            
            
   
# record_vehicles("inD",0)         
for n in range(2):
    record_vehicles("inD",n)

# for n in range(24):
#     record_vehicles("rounD",n)
#     print("rounD%d"%n)
    
# for n in range(13):
#     record_vehicles("uniD",n)
#     print("uniD%d"%n)
    
# for n in range(93):        
#     record_vehicles("exiD",n)
#     print("exiD%d"%n)
    


