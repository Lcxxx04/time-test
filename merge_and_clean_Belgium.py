import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 设置路径
# ==========================================
raw_data_path = r'D:\dataset\Belgium\energydata_complete.csv'  # 原始数据集路径
output_path = r'D:\Projects\Time-Series-Library-main\dataset\Belgium\Belgium_Final.csv'

# ==========================================
# 2. 读取数据
# ==========================================
print(f"📖 正在读取原始数据集: {raw_data_path}")
if not os.path.exists(raw_data_path):
    print("❌ 错误：找不到原始文件！")
    exit()

df = pd.read_csv(raw_data_path)
df['date'] = pd.to_datetime(df['date'])

# ==========================================
# 3. 变量提取与初步合并
# ==========================================
print("🧹 正在检查列名并合并变量...")

# 自动处理列名中的空格
df.columns = df.columns.str.strip()

# 打印所有列名，方便调试（如果还报错，看一眼控制台输出）
# print(f"当前数据集包含的列: {df.columns.tolist()}")

# 总用电量 WHE = Appliances + lights
df['WHE'] = df['Appliances'] + df['lights']

# 针对 To 可能被命名为 T_out 的兼容处理
temp_out_col = 'To' if 'To' in df.columns else 'T_out'

if temp_out_col not in df.columns:
    print(f"❌ 错误：找不到室外温度列！可用列名为: {df.columns.tolist()}")
    exit()

# 提取核心变量并统一重命名
df_subset = df[['date', 'WHE', temp_out_col, 'RH_out', 'Windspeed']].copy()
df_subset.rename(columns={
    temp_out_col: 'OT',
    'RH_out': 'RH',
    'Windspeed': 'Wind'
}, inplace=True)

# ==========================================
# 4. 频率转换 (10min -> 1h)
# ==========================================
print("⏳ 正在执行重采样 (10min -> 1h)...")
df_subset.set_index('date', inplace=True)

# 使用 mean 进行小时级聚合，保持数值量级的一致性
df_hourly = df_subset.resample('1h').mean()

# 处理重采样可能产生的空时间步（Gap）
df_hourly = df_hourly.interpolate(method='linear')

# ==========================================
# 5. 显式清洗 0 值与异常值
# ==========================================
# 1. WHE: 0 是故障
# 2. RH: 0 是故障
# 3. OT & Wind: 0 正常，不清洗

target_cols = ['WHE', 'RH']
print(f"\n🛠️ 开始清洗 0 值与异常值 (目标: {target_cols})...")

for col in target_cols:
    # 统计 0 值
    zeros = (df_hourly[col] == 0).sum()

    # 统计异常高值 (以 WHE 为例，使用 3-Sigma)
    if col == 'WHE':
        mean, std = df_hourly[col].mean(), df_hourly[col].std()
        upper_limit = mean + 3 * std
        outliers = (df_hourly[col] > upper_limit).sum()
    else:
        outliers = 0

    if zeros > 0 or outliers > 0:
        print(f"   🔍 [{col}] 发现 {zeros} 个 0 值, {outliers} 个异常高值 -> 正在插值修复...")
        # 将 0 和 异常高值 设为 NaN
        df_hourly.loc[df_hourly[col] == 0, col] = np.nan
        if col == 'WHE':
            df_hourly.loc[df_hourly[col] > upper_limit, col] = np.nan

        # 执行插值修复
        df_hourly[col] = df_hourly[col].interpolate(method='linear')
        df_hourly[col] = df_hourly[col].bfill().ffill()
    else:
        print(f"   ✅ [{col}] 很健康，无需处理。")

# ==========================================
# 6. 特别检查：温度 & 风速 (验证保留)
# ==========================================
print("\n🛡️ 正在验证保留变量 (OT, Wind)...")
print(f"   🌡️ 温度 (OT): 保留了 {(df_hourly['OT'] == 0).sum()} 个 0 值。")
print(f"   🌬️ 风速 (Wind): 保留了 {(df_hourly['Wind'] == 0).sum()} 个 0 值。")

# ==========================================
# 7. 保存结果
# ==========================================
df_final = df_hourly.reset_index()
print(f"\n💾 正在保存多变量数据集: {output_path}")
df_final.to_csv(output_path, index=False)

print("-" * 30)
print("🎉 比利时数据集小时化处理全部完成！")
print(f"最终变量: {df_final.columns.tolist()}")
print(f"数据总行数: {len(df_final)}")
print("-" * 30)