import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 设置路径
# ==========================================
elec_agg_path = r'D:\Projects\Time-Series-Library-main\dataset\London\Block3.csv'
weather_raw_path = r'D:\dataset\Smart meters in London\weather_hourly_darksky.csv'
output_path = r'D:\Projects\Time-Series-Library-main\dataset\London\Block3_Final_1h.csv'

# ==========================================
# 2. 读取电量数据并重采样 (30min -> 1h)
# ==========================================
print(f"📖 正在读取并重采样电量数据至 1h...")
df_elec = pd.read_csv(elec_agg_path)
df_elec['date'] = pd.to_datetime(df_elec['date'])
df_elec.set_index('date', inplace=True)

# 对电量进行 1 小时聚合。
df_elec_1h = df_elec.resample('1h').mean()
df_elec_1h = df_elec_1h.reset_index()

# ==========================================
# 3. 处理天气数据 (直接使用 1h 原始数据)
# ==========================================
print("🌡️ 正在处理天气数据 (对齐 1h)...")
df_weather = pd.read_csv(weather_raw_path)
df_weather['date'] = pd.to_datetime(df_weather['time'])

# 直接提取特征，无需插值
target_weather_cols = ['date', 'temperature', 'humidity', 'windSpeed']
df_weather_final = df_weather[target_weather_cols].copy()

# 统一重命名
df_weather_final.rename(columns={
    'temperature': 'OT',
    'humidity': 'RH',
    'windSpeed': 'Wind'
}, inplace=True)

# ==========================================
# 4. 合并 (Merge)
# ==========================================
print("🔗 正在执行 1h 多变量对齐合并...")
# 此时两者都是 1h 频率
df_merged = pd.merge(df_elec_1h, df_weather_final, on='date', how='inner')

# ==========================================
# 5. 显式清洗 0 值
# ==========================================
# 1. WHE (用电): 0 通常是故障，需要清洗
# 2. RH (湿度): 0 通常是故障，需要清洗
# 3. OT (温度): 0 是正常的 (0°C)，【不清洗】
# 4. Wind (风速): 0 是正常的 (无风)，【不清洗】

target_cols = ['WHE', 'RH']
print(f"\n🛠️ 开始清洗 0 值 (目标: {target_cols})...")
cleaned_count = 0

for col in target_cols:
    if col in df_merged.columns:
        zeros = (df_merged[col] == 0).sum()
        if zeros > 0:
            print(f"   🔍 [{col}] 发现 {zeros} 个 0 值 -> 视为异常，正在插值修复...")
            # 执行修复
            df_merged[col] = df_merged[col].replace(0, np.nan)
            df_merged[col] = df_merged[col].interpolate(method='linear')
            df_merged[col] = df_merged[col].bfill()
            df_merged[col] = df_merged[col].ffill()
            cleaned_count += 1
        else:
            print(f"   ✅ [{col}] 很健康，没有发现 0 值。")

# ==========================================
# 6. 特别检查：温度 & 风速 (验证保留)
# ==========================================
print("\n🛡️ 正在验证保留变量 (OT, Wind)...")

if 'OT' in df_merged.columns:
    temp_zeros = (df_merged['OT'] == 0).sum()
    print(f"   🌡️ 温度 (OT): 保留了 {temp_zeros} 个 0 值 (0°C 正常)。")

if 'Wind' in df_merged.columns:
    wind_zeros = (df_merged['Wind'] == 0).sum()
    print(f"   🌬️ 风速 (Wind): 保留了 {wind_zeros} 个 0 值 (无风/Calm 正常)。")

# ==========================================
# 7. 保存结果
# ==========================================
print(f"\n💾 正在保存多变量数据集: {output_path}")
df_merged.to_csv(output_path, index=False)

print("-" * 30)
print("🎉 处理流程全部完成！")
print(f"最终变量列表: {df_merged.columns.tolist()}")
print(f"数据总行数: {len(df_merged)}")
print("-" * 30)