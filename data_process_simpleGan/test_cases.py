import numpy as np
import pandas as pd
import os

# with open("testcases/inD/casescases.csv", "w") as f_cases:
#     f.truncate(0)

def trim_cases(Dtype,n):
    tracks_np = pd.read_csv("testcases/%(Dtype)s/%(Dtype)s_%(n)02d_testcases_vehicles_test.csv" %{'Dtype':Dtype,'Dtype':Dtype,'n':n}).values
    case_id=0
    row=0
    pre_id=2 #any num bigger than 1 is okay
    path="testcases/%(Dtype)s/cases/%(n)02d" %{'Dtype':Dtype, 'n':n}
    if not os.path.exists(path):
        os.makedirs(path)
        
    while row < len(tracks_np):
        if(int(tracks_np[row][0])<pre_id):
            case_id+=1
            with open("testcases/%(Dtype)s/cases/%(n)02d/%(case_id)03d_testcases.csv" %{'Dtype':Dtype,'n':n,'case_id':case_id}, "a") as f_cases:
                f_cases.truncate(0)
                f_cases.write("trackID,frame,x,y\n")
        with open("testcases/%(Dtype)s/cases/%(n)02d/%(case_id)03d_testcases.csv" %{'Dtype':Dtype,'n':n,'case_id':case_id}, "a") as f_cases:
            #start_frame=tracks_np[row][1]
            f_cases.write("%(id)d,%(frame)d,%(x)f,%(y)f\n" %{'id':int(tracks_np[row][0]),'frame':int(tracks_np[row][1]),'x':float(tracks_np[row][2]),'y':float(tracks_np[row][3])})
        pre_id=int(tracks_np[row][0])
        row+=1


# trim_cases("inD",0)         
for n in range(33):
    trim_cases("inD",n)