import pandas as pd
import os

# ==========================================
# 1. 设置路径
# ==========================================
input_file = r'D:\dataset\Smart meters in London\halfhourly_dataset\block_3.csv'
output_file = r'D:\Projects\Time-Series-Library-main\dataset\London\Block3.csv'

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# ==========================================
# 2. 读取电量数据
# ==========================================
print(f"📖 正在读取原始电量数据 (Block 0): {input_file}")
# 只读取需要的列以节省内存
# 假设列名是: ['LCLid', 'tstamp', 'KWh/hh']
df = pd.read_csv(input_file)

# 自动检测列名逻辑
time_col = 'tstamp' if 'tstamp' in df.columns else ('DateTime' if 'DateTime' in df.columns else df.columns[1])
load_col = 'KWh/hh' if 'KWh/hh' in df.columns else ('energy(kWh/hh)' if 'energy(kWh/hh)' in df.columns else df.columns[2])

print(f"   检测到时间列: {time_col}, 电量列: {load_col}")

# ==========================================
# 3. 聚合逻辑 (核心)
# ==========================================
print("⚡ 正在执行数据聚合 (按时间点取所有住户的平均值)...")

# 1. 转换时间格式
df['date'] = pd.to_datetime(df[time_col])

# 2. 强制转换电量为数值 (处理 'Null' 等异常)
df[load_col] = pd.to_numeric(df[load_col], errors='coerce')

# 3. 按时间分组并聚合
# 我们取均值(mean)，这样代表了伦敦该区域“平均每户”的水平
df_agg = df.groupby('date')[load_col].mean().reset_index()

# 4. 重命名
df_agg.rename(columns={load_col: 'WHE'}, inplace=True)

# ==========================================
# 4. 保存结果
# ==========================================
print(f"📊 聚合完成:")
print(f"   原始总行数: {len(df)}")
print(f"   聚合后行数: {len(df_agg)} (唯一时间点数量)")
print(f"💾 正在保存: {output_file}")

df_agg.to_csv(output_file, index=False)
print("✅ 处理成功！")