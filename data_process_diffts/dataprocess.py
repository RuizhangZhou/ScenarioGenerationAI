import numpy as np
import pandas as pd
import os
import math

def process_dataset(Dtype, n, seq_length=1000, interval=100):
    
    recordingMeta_file_path = "/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_recordingMeta.csv" %{'Dtype':Dtype,'n':n}
    df = pd.read_csv(recordingMeta_file_path)
    num_frames_recording=df["frameRate"].iloc[0]*df["duration"].iloc[0]
    print(num_frames_recording)

    # Calculate the number of cases
    num_cases = math.floor((num_frames_recording - seq_length) / interval) + 1

    # Initialize the list to store track IDs for each case
    trackIDLists = [list() for _ in range(num_cases)]

    # Load the CSV file
    csv_file_path = "/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_tracksMeta.csv" %{'Dtype':Dtype,'n':n}
    df = pd.read_csv(csv_file_path)

    # Iterate through each track
    for index, row in df.iterrows():
        initialFrame = row["initialFrame"]
        finalFrame = row["finalFrame"]
        numFrames = row["numFrames"]
        
        # Skip tracks with too many frames (optional criterion)
        if numFrames > 1500:
            continue
        
        # Calculate the start and end case indexes for the current vehicle
        start_case_idx = max(0, math.ceil((initialFrame - seq_length) / interval))
        end_case_idx = min(num_cases - 1, math.floor(finalFrame / interval))
        
        # Add the track ID to the relevant cases
        for i in range(start_case_idx, end_case_idx + 1):
            trackIDLists[i].append(row["trackId"])

    # # Print the result
    # for i, trackIDs in enumerate(trackIDLists):
    #     print(f"Case {i}: Track IDs: {trackIDs}")
        
    # 初始化一个列表来存储每个case的track数量
    track_counts = [len(track_ids) for track_ids in trackIDLists]

    # 计算最多的track数量
    max_tracks = max(track_counts)
    print(f"最多的track数量为: {max_tracks}")

    # 创建空DataFrame
    columns = ['caseID'] + [f'{xy}{i+1}' for i in range(max_tracks) for xy in ['x', 'y']]
    df = pd.DataFrame(columns=columns)

    # 准备track数据
    tracks_df = pd.read_csv("/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_tracks.csv" %{'Dtype':Dtype,'n':n})

    rows_list = []
    for case_idx in range(num_cases):
        print(f"case: {case_idx+1}")
        start_frame = case_idx * interval
        for frame in range(start_frame, start_frame + seq_length):
            row = {'caseID': case_idx + 1}  # caseID从1开始编号
            for track_id in trackIDLists[case_idx]:
                track_data = tracks_df[(tracks_df['trackId'] == track_id) & (tracks_df['frame'] == frame)]
                if not track_data.empty:
                    x_center = track_data['xCenter'].values[0]
                    y_center = track_data['yCenter'].values[0]
                else:
                    x_center, y_center = 0, 0  # 如果在当前frame没有track数据，则填充0
                # 请注意这里直接更新row字典
                row[f'x{track_id}'] = x_center
                row[f'y{track_id}'] = y_center
            rows_list.append(row)

    # 一次性将rows_list转换为DataFrame
    new_rows_df = pd.DataFrame(rows_list)

    # 现在可以将new_rows_df添加到原始df中
    df = pd.concat([df, new_rows_df], ignore_index=True)

    num_features=2*max_tracks
    output_file_path = "/DATA1/rzhou/ika/multi_testcases/%(Dtype)s_map%(n)02d_interval%(interval)d_seq%(seq_length)d_nfea%(num_features)d.csv" %{'Dtype':Dtype,'n':n, 'interval': interval, 'seq_length': seq_length, 'num_features': num_features}
    # 清空文件
    with open(output_file_path, 'w') as file:
        pass

    # 将DataFrame写入CSV
    df.to_csv("/DATA1/rzhou/ika/multi_testcases/%(Dtype)s/%(Dtype)s_map%(n)02d_interval%(interval)d_seq%(seq_length)d_nfea%(num_features)d.csv" %{'Dtype':Dtype,'n':n, 'interval': interval, 'seq_length': seq_length, 'num_features': num_features}, index=False)



# Loop through all inD and rounD datasets
# for Dtype in ["inD", "rounD"]:
#     num_files = {"inD": 32, "rounD": 8}[Dtype]
#     for n in range(num_files + 1):
#         print(f"Processing file {n} of {num_files} for {Dtype}")
#         process_dataset(Dtype, n)
process_dataset("inD",19)
        
#nohup python dataprocess.py >> /home/rzhou/Projects/scenariogenerationai/data_process_diffts/log/dataprocess.log 2>&1