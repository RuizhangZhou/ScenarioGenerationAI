import pandas as pd
import numpy as np

# 假设df是你的DataFrame，已经加载了csv文件
df = pd.read_csv('/DATA1/rzhou/ika/multi_testcases/rounD/ori/seq500/09-23/int10/rounD_map09-23_interval10_seq500_nfea10_pad0.csv')

seq_length = 500  # 定义每个case的长度

# 对于df中除了'caseID'以外的每一列
for col in df.columns:
    if col != 'caseID':
        # 对每个500行分段进行循环处理
        for start in range(0, len(df), seq_length):
            print(f"Processing {col} from {start} to {start + seq_length}")
            end = start + seq_length
            
            # 获取当前分段的数据
            segment = df.loc[start:end, col]
            
            # 寻找第一个非0值的索引
            first_non_zero_idx = segment.ne(0).idxmax()
            if segment[first_non_zero_idx] == 0:  # 如果整个段都是0，则跳过
                continue
            
            # 替换段起始的0值
            df.loc[start:first_non_zero_idx, col] = df.loc[first_non_zero_idx, col]
            
            # 寻找最后一个非0值的索引
            last_non_zero_idx = segment.ne(0)[::-1].idxmax()
            
            # 替换段结束的0值
            df.loc[last_non_zero_idx:end, col] = df.loc[last_non_zero_idx, col]

# 输出到新的CSV文件
output_file_path = "/DATA1/rzhou/ika/multi_testcases/rounD/ori/seq500/09-23/int10/rounD_map09-23_interval10_seq500_nfea10_padreal.csv"
df.to_csv(output_file_path, index=False)


# nohup python padding_value.py >> /home/rzhou/Projects/scenariogenerationai/data_process_diffts/log/padding_value.log 2>&1 &