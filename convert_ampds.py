import tables
import pandas as pd
import numpy as np
import os

# 1. 设置路径
h5_file_path = r'D:\dataset\AMPds2.h5'
output_csv_path = './dataset/AMPds/AMPds.csv'

os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
print(f"🔍 正在读取: {h5_file_path}")

try:
    with tables.open_file(h5_file_path, mode='r') as f:

        # ==========================================
        # 1. 处理电力数据 (Meter 1 - WHE)
        # ==========================================
        print("\n⚡ 正在解析电力数据...")
        # 获取节点
        table_elec = f.get_node('/building1/elec/meter1/table')
        raw_data = table_elec.read()

        # 核心解包逻辑
        if 'values_block_0' in raw_data.dtype.names:
            print("📦 检测到 Block 压缩格式，正在解包...")
            elec_matrix = raw_data['values_block_0']
            df_elec = pd.DataFrame(elec_matrix)
            print(f"✅ 电力数据解包成功! 形状: {df_elec.shape}")
        else:
            df_elec = pd.DataFrame(raw_data)

        # ==========================================
        # 2. 提取关键列 (WHE)
        # ==========================================
        print("\n🔍 正在锁定关键列...")
        result = pd.DataFrame()

        # 生成标准时间 (AMPds2 固定从 2012-04-01 开始)
        result['date'] = pd.date_range(start='2012-04-01 00:00:00', periods=len(df_elec), freq='1min')

        # 提取第 6 列 (索引 5) 作为有功功率 (WHE)
        # 根据 AMPds 文档和你的数据形状 (12列)，通常第 6 列是 Real Power
        target_idx = 5
        if df_elec.shape[1] > target_idx:
            print(f"正在提取第 {target_idx} 列作为 WHE (Real Power)...")
            result['WHE'] = df_elec.iloc[:, target_idx]
        else:
            result['WHE'] = df_elec.iloc[:, 0]

        # ==========================================
        # 3. 降采样与保存 (关键修复: 1H -> 1h)
        # ==========================================
        print("\n📉 正在降采样为小时级 (1h)...")
        result.set_index('date', inplace=True)

        # 修复点：将 '1H' 改为 '1h' 以兼容新版 Pandas
        result = result.resample('1h').mean().ffill()

        result.reset_index(inplace=True)

        print(f"💾 正在保存到: {output_csv_path}")
        result.to_csv(output_csv_path, index=False)
        print("🎉 恭喜！所有障碍已清除，数据转换完成！")
        print(result.head())

except Exception as e:
    print(f"\n❌ 出错: {e}")
    import traceback

    traceback.print_exc()