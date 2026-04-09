import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 设置文件路径
# ==========================================
elec_path = './dataset/AMPds/AMPds.csv'
weather_path = r'D:\dataset\AMPds\Climate_HourlyWeather.csv'
output_path = './dataset/AMPds/AMPds_Final.csv'

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ==========================================
# 2. 读取数据
# ==========================================
print(f"📖 正在读取电量数据: {elec_path}")
df_elec = pd.read_csv(elec_path)

print(f"📖 正在读取天气数据: {weather_path}")
df_weather = pd.read_csv(weather_path)

# ==========================================
# 3. 处理时间戳
# ==========================================
print("\n⏳ 正在对齐时间格式...")
if 'date' not in df_elec.columns:
    time_col_elec = df_elec.columns[0]
    df_elec.rename(columns={time_col_elec: 'date'}, inplace=True)
df_elec['date'] = pd.to_datetime(df_elec['date'])

time_col_weather = df_weather.columns[0]
df_weather.rename(columns={time_col_weather: 'date'}, inplace=True)
df_weather['date'] = pd.to_datetime(df_weather['date'])

# ==========================================
# 4. 智能筛选天气数值列 (OT, RH, Wind)
# ==========================================
print("✂️ 正在筛选天气数值列 (OT, RH, Wind)...")

weather_final = pd.DataFrame()
weather_final['date'] = df_weather['date']


def is_valid_data_col(col_name, keyword):
    c_lower = col_name.lower()
    if keyword not in c_lower: return False
    if 'flag' in c_lower or 'qual' in c_lower or 'desc' in c_lower: return False
    return True


found_temp, found_hum, found_wind = False, False, False

for col in df_weather.columns:
    c_lower = col.lower()
    if is_valid_data_col(col, 'temp') and 'dew' not in c_lower and not found_temp:
        print(f"   ✅ 锁定温度: [{col}] -> OT")
        weather_final['OT'] = pd.to_numeric(df_weather[col], errors='coerce')
        found_temp = True
    elif is_valid_data_col(col, 'hum') and not found_hum:
        print(f"   ✅ 锁定湿度: [{col}] -> RH")
        weather_final['RH'] = pd.to_numeric(df_weather[col], errors='coerce')
        found_hum = True
    elif is_valid_data_col(col, 'wind') and ('spd' in c_lower or 'speed' in c_lower):
        if 'chill' not in c_lower and not found_wind:
            print(f"   ✅ 锁定风速: [{col}] -> Wind")
            weather_final['Wind'] = pd.to_numeric(df_weather[col], errors='coerce')
            found_wind = True

# ==========================================
# 5. 合并与初步修复 NaN
# ==========================================
print("\n🔗 正在合并数据并修复 NaN (由 'Null' 或缺失引起)...")
weather_final.set_index('date', inplace=True)
weather_final = weather_final.interpolate(method='time').ffill().bfill()
weather_final.reset_index(inplace=True)

df_merged = pd.merge(df_elec, weather_final, on='date', how='left')
df_merged.ffill(inplace=True)
df_merged.bfill(inplace=True)

# ==========================================
# 6. 【新增】深度诊断与 0 值清洗
# ==========================================
print("\n🛠️ 开始清洗 0 值与异常值 (WHE, RH)...")

# 寻找电量列
load_col = 'WHE' if 'WHE' in df_merged.columns else df_merged.columns[1]
target_cols = [load_col, 'RH']

for col in target_cols:
    if col in df_merged.columns:
        # 统计
        zeros = (df_merged[col] == 0).sum()
        nans = df_merged[col].isna().sum()

        if zeros > 0 or nans > 0:
            print(f"   🔍 [{col}] 发现 {zeros} 个 0 值, {nans} 个 NaN -> 正在执行修复...")
            df_merged[col] = df_merged[col].replace(0, np.nan)
            df_merged[col] = df_merged[col].interpolate(method='linear').ffill().bfill()
        else:
            print(f"   ✅ [{col}] 数据很健康，无异常 0 值。")

# 验证保留项
print("\n🛡️ 正在验证保留变量 (OT, Wind)...")
if 'OT' in df_merged.columns:
    print(f"   🌡️ 温度 (OT): 保留了 {(df_merged['OT'] == 0).sum()} 个 0°C 点。")
if 'Wind' in df_merged.columns:
    print(f"   🌬️ 风速 (Wind): 保留了 {(df_merged['Wind'] == 0).sum()} 个静风点。")

# ==========================================
# 7. 保存结果
# ==========================================
print(f"\n💾 正在保存最终数据集: {output_path}")
df_merged.to_csv(output_path, index=False)

print("-" * 30)
print("🎉 AMPds 处理流水线全部完成！")
print(f"最终变量列表: {df_merged.columns.tolist()}")
print(df_merged.head())
print("-" * 30)