import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 设置路径
# ==========================================
raw_data_path = r'D:\dataset\Mexico\energy_weather_raw_data.csv'
output_path = r'D:\Projects\Time-Series-Library-main\dataset\Mexico\Mexico_Final.csv'

# ==========================================
# 2. 读取数据
# ==========================================
print(f"📖 正在读取墨西哥原始数据集: {raw_data_path}")
if not os.path.exists(raw_data_path):
    print("❌ 错误：找不到原始文件！请检查路径。")
    exit()

df = pd.read_csv(raw_data_path)

# 自动识别时间列并转换
if 'datetime' in df.columns:
    df['date'] = pd.to_datetime(df['datetime'])
elif 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
else:
    # 如果没有时间列，尝试将第一列设为时间
    df['date'] = pd.to_datetime(df.iloc[:, 0])

# ==========================================
# 3. 变量提取与列名对齐
# ==========================================
print("🧹 正在映射墨西哥数据集列名...")

# 墨西哥数据集列名映射表
# 原始名 -> 目标名
# active_power -> WHE (总用电)
# temp -> OT (室外温度)
# humidity -> RH (相对湿度)
# speed -> Wind (风速)

mapping = {
    'active_power': 'WHE',
    'temp': 'OT',
    'humidity': 'RH',
    'speed': 'Wind'
}

# 检查列是否存在并提取
available_cols = [col for col in mapping.keys() if col in df.columns]
if len(available_cols) < 4:
    print(f"⚠️ 警告：缺少部分预期列！当前找到: {available_cols}")

df_subset = df[['date'] + available_cols].copy()
df_subset.rename(columns=mapping, inplace=True)

# ==========================================
# 4. 频率转换 (1min -> 1h)
# ==========================================
print("⏳ 正在执行重采样 (1min -> 1h)... 墨西哥数据量大，请稍候...")
df_subset.set_index('date', inplace=True)

# 使用 mean 聚合，'1h' 适配新版 Pandas
df_hourly = df_subset.resample('1h').mean()

# 处理重采样可能产生的缺失小时 (Gap)
df_hourly = df_hourly.interpolate(method='linear')

# ==========================================
# 5. 显式清洗 0 值与异常值 (3-Sigma)
# ==========================================
target_cols = ['WHE', 'RH']
print(f"\n🛠️ 开始清洗 0 值与异常值 (目标: {target_cols})...")

for col in target_cols:
    if col not in df_hourly.columns: continue

    # 1. 统计 0 值
    zeros = (df_hourly[col] == 0).sum()

    # 2. 统计异常高值 (使用 3-Sigma 准则)
    mean, std = df_hourly[col].mean(), df_hourly[col].std()
    upper_limit = mean + 3 * std
    outliers = (df_hourly[col] > upper_limit).sum()

    if zeros > 0 or outliers > 0:
        print(f"   🔍 [{col}] 发现 {zeros} 个 0 值, {outliers} 个异常高值(>{upper_limit:.2f}) -> 正在插值修复...")

        # 将 0 和 异常高值 设为 NaN 方便插值
        df_hourly.loc[df_hourly[col] == 0, col] = np.nan
        df_hourly.loc[df_hourly[col] > upper_limit, col] = np.nan

        # 执行插值修复
        df_hourly[col] = df_hourly[col].interpolate(method='linear')
        # 边缘情况处理
        df_hourly[col] = df_hourly[col].bfill().ffill()
    else:
        print(f"   ✅ [{col}] 健康度良好。")

# ==========================================
# 6. 特别检查：验证保留
# ==========================================
print("\n🛡️ 验证环境指标 (OT, Wind)...")
if 'OT' in df_hourly.columns:
    print(f"   🌡️ 温度 (OT): 范围 {df_hourly['OT'].min():.2f} ~ {df_hourly['OT'].max():.2f}")
if 'Wind' in df_hourly.columns:
    print(f"   🌬️ 风速 (Wind): 范围 {df_hourly['Wind'].min():.2f} ~ {df_hourly['Wind'].max():.2f}")

# ==========================================
# 7. 保存结果
# ==========================================
df_final = df_hourly.reset_index()

# 确保保存目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print(f"\n💾 正在保存墨西哥多变量数据集: {output_path}")
df_final.to_csv(output_path, index=False)

print("-" * 30)
print("🎉 墨西哥数据集预处理完成！")
print(f"数据周期: {df_final['date'].min()} 至 {df_final['date'].max()}")
print(f"最终变量: {df_final.columns.tolist()}")
print(f"数据总行数: {len(df_final)}")
print("-" * 30)