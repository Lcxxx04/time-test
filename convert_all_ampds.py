import tables
import pandas as pd
import numpy as np
import os

# 1. 设置路径
h5_file_path = r'D:\dataset\AMPds2.h5'
output_csv_path = './dataset/AMPds/AMPds_Submeters.csv'

os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
print(f"🔍 正在全量扫描文件 (过滤模式): {h5_file_path}")

# 变量名映射
name_map = {
    'meter1': 'WHE',  # 全屋总电量
    'meter2': 'CDE',  # 干衣机
    'meter3': 'CWE',  # 洗衣机
    'meter4': 'DWE',  # 洗碗机
    'meter5': 'EQE',  # 电子设备
    'meter6': 'FGE',  # 冰箱
    'meter7': 'FRE',  # 备用冰箱
    'meter8': 'HPE',  # 热泵/空调
    'meter9': 'HPE_g',  # 车库空调
    'meter10': 'KWE',  # 厨房插座
    'meter11': 'LWE',  # 客厅插座
}

try:
    with tables.open_file(h5_file_path, mode='r') as f:

        all_series = {}
        # AMPds2 标准长度 (2年分钟级数据)
        EXPECTED_LEN = 1051200

        print("\n🚀 开始遍历数据表...")

        for node in f.walk_nodes('/', classname='Table'):
            path = node._v_pathname

            # ：跳过 cache 文件夹
            if '/cache/' in path:
                continue

            # 只关心 meter 数据
            if 'elec' in path and 'meter' in path:
                # 提取 meter 名称
                meter_name = path.split('/')[-2]  # 获取 meterX

                # 确定变量名
                var_name = name_map.get(meter_name, meter_name)

                # 读取数据
                raw_data = node.read()

                # 解包 Logic
                if 'values_block_0' in raw_data.dtype.names:
                    data_matrix = raw_data['values_block_0']
                    # 提取 Real Power (通常是第 6 列, index 5)
                    target_idx = 5 if data_matrix.shape[1] > 5 else 0
                    power_data = data_matrix[:, target_idx]

                    # 长度检查
                    if len(power_data) == EXPECTED_LEN:
                        all_series[var_name] = power_data
                        print(f"   ✅ 提取成功: {var_name} (长度: {len(power_data)})")
                    else:
                        print(f"   ⚠️ 跳过 {var_name}: 长度不一致 ({len(power_data)})")

        # 3. 合并
        print("\n📦 正在合并数据...")
        if not all_series:
            print("❌ 未提取到有效数据！")
            exit()

        df_all = pd.DataFrame(all_series)

        # 生成时间轴
        print("⏳ 生成时间轴...")
        df_all['date'] = pd.date_range(start='2012-04-01 00:00:00', periods=EXPECTED_LEN, freq='1min')

        # 4. 降采样 (1h)
        print("📉 正在降采样为小时级 (1h)...")
        df_all.set_index('date', inplace=True)
        df_resampled = df_all.resample('1h').mean().ffill()
        df_resampled.reset_index(inplace=True)

        # 5. 保存
        print(f"💾 保存到: {output_csv_path}")
        df_resampled.to_csv(output_csv_path, index=False)
        print("🎉 全部完成！你的数据集包含以下变量：")
        print(df_resampled.columns.tolist())
        print(df_resampled.head())

except Exception as e:
    print(f"\n❌ 出错: {e}")
    import traceback

    traceback.print_exc()