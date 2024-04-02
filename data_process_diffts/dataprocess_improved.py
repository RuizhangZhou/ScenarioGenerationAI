import pandas as pd
import numpy as np
import math
import os
def optimize_process(Dtype, n, seq_length=500, interval=100):

    recordingMeta_file_path = "/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_recordingMeta.csv" %{'Dtype':Dtype,'n':n}
    recordingMeta_df = pd.read_csv(recordingMeta_file_path)
    num_frames_recording=recordingMeta_df["frameRate"].iloc[0]*recordingMeta_df["duration"].iloc[0]
    numTracks=recordingMeta_df["numTracks"].iloc[0]

    
    
    # 参数设置
    seq_length = 500  # 每个案例的帧数长度
    interval = 100  # 案例之间的开始帧数间隔

    num_cases = math.ceil((num_frames_recording - seq_length) / interval) + 1
    print(f"num_cases:{num_cases}")

    # 初始化列表存储每个案例的track IDs
    trackIDLists = [[] for _ in range(num_cases)]

    # 读取tracksMeta文件
    csv_file_path = "/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_tracksMeta.csv" % {'Dtype': Dtype, 'n': n}
    df_tracksMeta = pd.read_csv(csv_file_path)
    
    # Initialize an array to hold the pointer for each trackId
    pointers = np.zeros(numTracks, dtype=int)
    pointers[1:] = np.cumsum(df_tracksMeta['numFrames'].values)[:-1]

    # 遍历每个track，确定它出现在哪些案例中
    for index, row in df_tracksMeta.iterrows():
        initialFrame = row['initialFrame']
        finalFrame = row['finalFrame']
        numFrames = row['numFrames']

        # 跳过帧数过多的tracks
        if numFrames > 1500:
            continue

        # 确定track出现在哪些案例中
        for i in range(num_cases):
            case_start_frame = i * interval
            case_end_frame = case_start_frame + seq_length - 1
            # 检查track的帧区间是否与当前案例的帧区间有交集
            if not (finalFrame < case_start_frame or initialFrame > case_end_frame):
                trackIDLists[i].append(row['trackId'])

    # 筛选出非空的trackIDLists
    filtered_trackIDLists = [trackIDs for trackIDs in trackIDLists if trackIDs]
    num_cases_filtered=len(filtered_trackIDLists)

    # 打印结果
    # for i, trackIDs in enumerate(filtered_trackIDLists):
    #     print(f"Case {i+1}: Track IDs: {trackIDs}")

    # 初始化一个列表来存储每个案例的track数量，并计算最多的track数量
    track_counts = [len(track_ids) for track_ids in filtered_trackIDLists]
    max_tracks = max(track_counts)

    #print(f"最多的track数量为: {max_tracks}")
    # 准备track数据
    tracks_df = pd.read_csv("/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_tracks.csv" %{'Dtype':Dtype,'n':n})
    # 创建列名
    columns = ['caseID'] + [f'{xy}{i+1}' for i in range(max_tracks) for xy in ['x', 'y']]

    # 计算总行数
    total_rows = seq_length * num_cases_filtered

    # 创建一个全是0的DataFrame，指定float64类型
    df = pd.DataFrame(0, index=np.arange(total_rows), columns=columns).astype({'caseID': 'int'}).astype({col: 'float64' for col in columns if col != 'caseID'})

    # 设置caseID列
    df['caseID'] = np.repeat(np.arange(1, num_cases_filtered + 1), seq_length)
    print(f"num_cases_filtered:{num_cases_filtered}")

    # 遍历每个案例start_frame_of_case
    start_frame_of_case=0
    for case_id in range(num_cases_filtered):
        #start_frame_of_case = interval * case_id
        start_row_of_case = case_id * seq_length

        # 遍历当前案例中的轨迹ID
        # for track_idx in range(len(filtered_trackIDLists[case_id])):
        #     track_id = filtered_trackIDLists[case_id][track_idx]
        for track_idx, track_id in enumerate(trackIDLists[case_id]):
            # print(track_idx)
            # print(track_id)
            # 获取轨迹的初始帧和最终帧
            initial_frame = df_tracksMeta.loc[track_id, "initialFrame"]
            
            while start_frame_of_case + seq_length - 1 < initial_frame : start_frame_of_case += interval 
            # 如果当前case中的track的initialframe不在当前的start_frame_of_case ~ start_frame_of_case + seq_length-1中
            
            final_frame = df_tracksMeta.loc[track_id, "finalFrame"]
            
            # 确定当前案例中轨迹的起止帧
            start_frame = max(start_frame_of_case, initial_frame)
            end_frame = min(start_frame_of_case + seq_length - 1, final_frame)
            
            # 计算在df中对应的起止行
            start_row = start_row_of_case + (start_frame - start_frame_of_case)
            end_row = start_row_of_case + (end_frame - start_frame_of_case)
            
            # 从tracks文件中提取轨迹数据
            track_rows_start = pointers[track_id] + (start_frame - initial_frame)
            track_rows_end = track_rows_start + (end_frame - start_frame + 1)
            #track_rows_end=pointers[track_id]+end_frame-initial_frame
            
            track_data = tracks_df.iloc[track_rows_start:track_rows_end]
            if track_data.empty:
                print(f"case_id:{case_id}, track_id:{track_id}, track_rows_start:{track_rows_start},track_rows_end: {track_rows_end}, start_frame:{start_frame}, end_frame:{end_frame}")
                print("没有找到对应的track数据")
            
            # 更新DataFrame
            df.loc[start_row:end_row, f'x{track_idx+1}'] = track_data["xCenter"].values
            df.loc[start_row:end_row, f'y{track_idx+1}'] = track_data["yCenter"].values
            
        start_frame_of_case += interval

    # 注意：这里假设tracks_df是按trackId和frame排序的，而且pointers数组已正确初始化

    num_features=2*max_tracks

    output_file_path = "/DATA1/rzhou/ika/multi_testcases/%(Dtype)s/%(Dtype)s_map%(n)02d_interval%(interval)d_seq%(seq_length)d_nfea%(num_features)d.csv" %{'Dtype':Dtype,'n':n, 'interval': interval, 'seq_length': seq_length, 'num_features': num_features}
    # 清空文件
    with open(output_file_path, 'w') as file:
        pass

    # 将DataFrame写入CSV
    df.to_csv("/DATA1/rzhou/ika/multi_testcases/%(Dtype)s/%(Dtype)s_map%(n)02d_interval%(interval)d_seq%(seq_length)d_nfea%(num_features)d.csv" %{'Dtype':Dtype,'n':n, 'interval': interval, 'seq_length': seq_length, 'num_features': num_features}, index=False)




# Loop through all inD and rounD datasets
for Dtype in ["inD", "rounD"]:
    num_files = {"inD": 32, "rounD": 8}[Dtype]
    for n in range(num_files + 1):
        print(f"Processing file {n} of {num_files} for {Dtype}")
        optimize_process(Dtype, n)
        
# Dtype="rounD"
# for num_files in range(2,9):
#     print(f"Processing file {n} of {num_files} for {Dtype}")
#         optimize_process(Dtype, n)
        
        
# Example call
# optimize_process("inD", 4)



#nohup python dataprocess_improved.py >> /home/rzhou/Projects/scenariogenerationai/data_process_diffts/log/dataprocess_improved.log 2>&1