import numpy as np
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES']='2'


# with open("testcases/inD/casescases.csv", "w") as f_cases:
#     f.truncate(0)


# def trim_cases(Dtype,n):
#     tracks_np = pd.read_csv("testcases/%(Dtype)s/cases/%(n)02d" %{'Dtype':Dtype,'n':n}).values
#     case_id=0
#     row=0
#     pre_id=2 #any num bigger than 1 is okay
#     path="testcases/%(Dtype)s/cases/%(n)02d" %{'Dtype':Dtype, 'n':n}
#     if not os.path.exists(path):
#         os.makedirs(path)
        
#     while row < len(tracks_np):
#         if(int(tracks_np[row][0])<pre_id):
#             case_id+=1
#             with open("testcases/%(Dtype)s/cases/%(n)02d/%(case_id)03d_testcases.csv" %{'Dtype':Dtype,'n':n,'case_id':case_id}, "a") as f_cases:
#                 f_cases.truncate(0)
#                 f_cases.write("trackID,frame,x,y\n")
#         with open("testcases/%(Dtype)s/cases/%(n)02d/%(case_id)03d_testcases.csv" %{'Dtype':Dtype,'n':n,'case_id':case_id}, "a") as f_cases:
#             #start_frame=tracks_np[row][1]
#             f_cases.write("%(id)d,%(frame)d,%(x)f,%(y)f\n" %{'id':int(tracks_np[row][0]),'frame':int(tracks_np[row][1]),'x':float(tracks_np[row][2]),'y':float(tracks_np[row][3])})
#         pre_id=int(tracks_np[row][0])
#         row+=1


def getfiles(Dtype,n):
    filenames=os.listdir('/home/rzhou/Projects/generating-models-for-test-cases/scenariogenerationai/testcases/%(Dtype)s/cases/%(n)02d' %{'Dtype':Dtype,'n':n})
    # filenames=os.listdir('testcases/inD/cases/00' %{'Dtype':Dtype,'n':n})
    return filenames

def fulfill(Dtype,n):
    cases_files=getfiles(Dtype,n)
    cases_count=len(cases_files)
    case_id=1
    for case_file in cases_files:
        
        cases_np = pd.read_csv("/home/rzhou/Projects/generating-models-for-test-cases/scenariogenerationai/testcases/%(Dtype)s/cases/%(n)02d/%(case_file)s" %{'Dtype':Dtype,'n':n, 'case_file':case_file}).values
        with open("/home/rzhou/Projects/generating-models-for-test-cases/scenariogenerationai/testcases/%(Dtype)s/cases/%(n)02d/%(case_file)s_full.csv" %{'Dtype':Dtype,'n':n,'case_file':case_file}, "a") as f_cases_full:
            f_cases_full.truncate(0)
            f_cases_full.write("trackID,frame,x,y\n")
            row=0
            id=0
            while row<len(cases_np):
                if cases_np[row][0]!= id:
                    if cases_np[row][0]!= 1:
                        cur_frame=int(cases_np[row-1][1])
                        for f in range(cur_frame+1,1501):
                            f_cases_full.write("%(id)d,%(f)d,%(x)f,%(y)f\n" %{'id':id,'f':f,'x':0,'y':0})
                    id+=1
                    for f in range(0,int(cases_np[row][1])):
                        f_cases_full.write("%(id)d,%(f)d,%(x)f,%(y)f\n" %{'id':id,'f':f,'x':0,'y':0})
                        
                f_cases_full.write("%(id)d,%(f)d,%(x)f,%(y)f\n" %{'id':id,'f':int(cases_np[row][1]),'x':float(cases_np[row][2]),'y':float(cases_np[row][3])}) 
                row+=1
                
            cur_frame=int(cases_np[row-1][1])
            for f in range(cur_frame+1,1501):
                f_cases_full.write("%(id)d,%(f)d,%(x)f,%(y)f\n" %{'id':id,'f':f,'x':0,'y':0})


      
for n in range(33):
    fulfill("inD",n)
