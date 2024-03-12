import numpy as np
import pandas as pd
import os
import csv
import math

#exiD: 0-92
#highD: 1-60
#inD: 0-32
#rounD: 0-23
#uniD: 0-12
Dtype="uniD" 
for n in range(13):
    recordingMeta_np = pd.read_csv("/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_recordingMeta.csv" %{'Dtype':Dtype,'n':n}, low_memory=False).values
    if Dtype=="highD":
        numTracks_recordingMeta_np=recordingMeta_np[0][10]
    else:
        numTracks_recordingMeta_np=recordingMeta_np[0][7]
    tracks_np = pd.read_csv("/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_tracks.csv" %{'Dtype':Dtype,'n':n}, low_memory=False).values
    numTracks_tracks_np=tracks_np[-2][1]
    
    print(numTracks_recordingMeta_np)
    if Dtype=="highD":
        print(numTracks_tracks_np)
    else:
        print(numTracks_tracks_np+1)
    if Dtype=="highD":
        if(numTracks_recordingMeta_np!=numTracks_tracks_np): 
            print("%(Dtype)s %(n)02d data is wrong" %{'Dtype':Dtype,'n':n})
    else:
        if(numTracks_recordingMeta_np!=numTracks_tracks_np+1): 
            print("%(Dtype)s %(n)02d data is wrong" %{'Dtype':Dtype,'n':n})
