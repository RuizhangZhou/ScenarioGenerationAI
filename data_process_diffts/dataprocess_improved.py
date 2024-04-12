import pandas as pd
import numpy as np
import math
import os
def optimize_process(Dtype, n, seq_length=500, interval=50):

    recordingMeta_file_path = "/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_recordingMeta.csv" %{'Dtype':Dtype,'n':n}
    recordingMeta_df = pd.read_csv(recordingMeta_file_path)
    num_frames_recording=recordingMeta_df["frameRate"].iloc[0]*recordingMeta_df["duration"].iloc[0]
    numTracks=recordingMeta_df["numTracks"].iloc[0]

    # 参数设置
    seq_length = 500  # 每个案例的帧数长度
    interval = 10  # 案例之间的开始帧数间隔
    percentage=0.9  #track的帧区间与案例帧区间的交集长度占案例帧区间长度的比例  
    min_v=5   #至少要有n辆车

    # num_cases为
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
        initial_frame = row['initialFrame']
        final_frame = row['finalFrame']
        numFrames = row['numFrames']
        v_class= row['class']

        # 跳过帧数过多或过少的tracks,或者是行人
        if numFrames > 4*seq_length or v_class=="pedestrian" or v_class=="bicycle" or v_class=="motorcycle":
            continue

        # 确定track出现在哪些案例中
        for i in range(num_cases):
            case_start_frame = i * interval
            case_end_frame = case_start_frame + seq_length - 1
            # 检查track的帧区间是否与当前案例的帧区间有交集,并且交集长度大于seq_length的一半
            start_frame = max(case_start_frame, initial_frame)
            end_frame = min(case_start_frame + seq_length - 1, final_frame)
            if (not (final_frame < case_start_frame or initial_frame > case_end_frame) and end_frame-start_frame>=percentage*seq_length) \
            or (initial_frame >= case_start_frame and final_frame <= case_end_frame):
                trackIDLists[i].append(row['trackId'])
            
            # 检查track的帧区间是否完全包含在当前案例的帧区间内
            # if initial_frame >= case_start_frame and final_frame <= case_end_frame:
            #     trackIDLists[i].append(row['trackId'])

    # 筛选出非空的trackIDLists
    # filtered_trackIDLists = [trackIDs for trackIDs in trackIDLists if trackIDs]
    
    # 统计trackIDLists中len(trackIDs)出现次数最多的len(trackIDs),比如统计所有len(trackIDs)的分布,len(trackIDs)在trackIDLists中出现次数最多的len(trackIDs)
    # track_counts = [len(track_ids) for track_ids in trackIDLists]
    # print(f"track_counts:{track_counts}")
    # #统计track_counts中出现次数最多的元素
    # min_v = max(set(track_counts), key=track_counts.count)
    # print(f"min_v:{min_v}")
    
    # 创建一个num_cases长的boolean array,对应每个案例是否有min_v个以上的track, 并统计共有多少个这样的案例
    case_has_min_tracks = [len(trackIDs) >= min_v for trackIDs in trackIDLists]
    num_cases_has_min_tracks = sum(case_has_min_tracks)
    print(f"case_has_min_tracks:{case_has_min_tracks}")
    # trackIdLists中的trackIds如果大于min_v个,裁剪为前min_v个
    #trackIDLists = [trackIDs[:min_v] for trackIDs in trackIDLists]
    # 现在我的裁剪方式想为:如果trackIds大于min_v个, trackID的finalFrame-initialFrame最大的前min_v个trackIds
    trackIDLists = [sorted(trackIDs, key=lambda trackID: df_tracksMeta.loc[trackID, "finalFrame"] - df_tracksMeta.loc[trackID, "initialFrame"], reverse=True)[:min_v] for trackIDs in trackIDLists]

    # 打印结果
    for i, trackIDs in enumerate(trackIDLists):
        if len(trackIDs) >= min_v:
            print(f"Case {i+1}: Track IDs: {trackIDs}")

    # 初始化一个列表来存储每个案例的track数量，并计算最多的track数量
    # track_counts = [len(track_ids) for track_ids in filtered_trackIDLists]
    # max_tracks = max(track_counts)
    # min_tracks = min(track_counts)
    # print(f"n:{n}, 最少的track数量为: {min_tracks},最多的track数量为: {max_tracks}")
    # 准备track数据
    tracks_df = pd.read_csv("/DATA1/rzhou/ika/%(Dtype)s/data/%(n)02d_tracks.csv" %{'Dtype':Dtype,'n':n})
    # 创建列名
    columns = ['caseID'] + [f'{xy}{i+1}' for i in range(min_v) for xy in ['x', 'y']]

    # 计算总行数
    total_rows = seq_length * num_cases_has_min_tracks

    # 创建一个全是0的DataFrame，指定float64类型
    df = pd.DataFrame(0, index=np.arange(total_rows), columns=columns).astype({'caseID': 'int'}).astype({col: 'float64' for col in columns if col != 'caseID'})

    # 设置caseID列
    df['caseID'] = np.repeat(np.arange(1, num_cases_has_min_tracks + 1), seq_length)

    # 遍历每个案例start_frame_of_case
    case_idx=0
    for case_id in range(num_cases):
        if not case_has_min_tracks[case_id]: continue
        start_frame_of_case = interval * case_id
        start_row_of_case = case_idx * seq_length
        case_idx+=1
        # 遍历当前案例中的轨迹ID
        # for track_idx in range(len(filtered_trackIDLists[case_id])):
        #     track_id = filtered_trackIDLists[case_id][track_idx]
        for track_idx, track_id in enumerate(trackIDLists[case_id]):

            # 获取轨迹的初始帧和最终帧
            initial_frame = df_tracksMeta.loc[track_id, "initialFrame"]
            final_frame = df_tracksMeta.loc[track_id, "finalFrame"]
            
            # 确定当前案例中轨迹的起止帧
            start_frame = max(start_frame_of_case, initial_frame)
            end_frame = min(start_frame_of_case + seq_length - 1, final_frame)
            
            # 计算在df中对应的起止行
            start_row = start_row_of_case + (start_frame - start_frame_of_case)
            end_row = start_row_of_case + (end_frame - start_frame_of_case)
            
            # 从tracks文件中提取轨迹数据
            track_rows_start = pointers[track_id] + (start_frame - initial_frame)
            track_rows_end = track_rows_start + (end_frame - start_frame )
            #track_rows_end=pointers[track_id]+end_frame-initial_frame
            
            track_data = tracks_df.iloc[track_rows_start:track_rows_end + 1 ]#实际上是截取的track_rows_start到track_rows_end
            
            # 更新DataFrame
            df.loc[start_row:end_row, f'x{track_idx+1}'] = track_data["xCenter"].values
            df.loc[start_row:end_row, f'y{track_idx+1}'] = track_data["yCenter"].values
            
        #start_frame_of_case += interval

    # 注意：这里假设tracks_df是按trackId和frame排序的，而且pointers数组已正确初始化

    num_features=2*min_v

    def checkDirExistOrCreate(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    # 定义文件路径
    dtypeDir = f'/DATA1/rzhou/ika/multi_testcases/{Dtype}'
    checkDirExistOrCreate(dtypeDir)
    cases_csv_path = f'{dtypeDir}/{Dtype}_map{n:02d}_interval{interval}_seq{seq_length}_nfea{num_features}.csv'
    
    
    
    # 清空文件
    with open(cases_csv_path, 'w') as file:
        pass

    # 将DataFrame写入CSV
    df.to_csv(cases_csv_path, index=False)




# Loop through all inD and rounD datasets
# for Dtype in ["inD", "rounD"]:
#     num_files = {"inD": 32, "rounD": 8}[Dtype]
#     for n in range(num_files + 1):
#         print(f"Processing file {n} of {num_files} for {Dtype}")
#         optimize_process(Dtype, n)
        
Dtype="exiD"
for num_files in range(0,19):
    print(f"Processing file {num_files} of {num_files} for {Dtype}")
    optimize_process(Dtype, num_files)
        
        
# Example call
#optimize_process("inD", 29)



#nohup python dataprocess_improved.py >> /home/rzhou/Projects/scenariogenerationai/data_process_diffts/log/dataprocess_improved.log 2>&1 &