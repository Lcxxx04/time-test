import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# -----------------------
# 1) 环境检测
# -----------------------
print("=" * 60)
print(f"🔍 Python: {sys.executable}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# -----------------------
# 2) 参数配置
# -----------------------
DATA_PATH = "dataset/London/Block1_Final_1h.csv"
TARGET_COL_NAME = "WHE"

SEQ_LEN = 512
PRED_LEN = 96
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-4
PATIENCE = 10

# 三段切分
TRAIN_RATIO = 0.7
VALI_RATIO = 0.1
TEST_RATIO = 0.2

dataset_name = "LondonB1"
setting = f"long_term_forecast_{dataset_name}_{SEQ_LEN}_{PRED_LEN}_CNN_LSTM_custom_ftMS_sl{SEQ_LEN}_ll48_pl{PRED_LEN}_Exp_0"

checkpoints_dir = os.path.join("./checkpoints", setting)
results_dir = os.path.join("./results", setting)
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
model_save_path = os.path.join(checkpoints_dir, "checkpoint.pth")

# -----------------------
# 3) 辅助函数
# -----------------------
def masked_mape(preds, trues, eps=1e-6):
    denom = np.maximum(np.abs(trues), eps)
    return np.mean(np.abs((trues - preds) / denom)) * 100

class EarlyStopping:
    def __init__(self, save_path, patience=7):
        self.save_path = save_path
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(model)
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(model)
            self.counter = 0

    def _save(self, model):
        torch.save(model.state_dict(), self.save_path)

def create_sequences(x_block, y_block, seq_len, pred_len):
    """
    x_block: (T, D) scaled features
    y_block: (T, 1) scaled target (STANDARDIZED)
    return:
      X: (N, seq_len, D)
      y: (N, pred_len, 1)
    """
    xs, ys = [], []
    T = len(x_block)
    for i in tqdm(range(T - seq_len - pred_len + 1), desc="生成序列"):
        xs.append(x_block[i:i + seq_len, :])
        ys.append(y_block[i + seq_len:i + seq_len + pred_len, 0])
    X = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)[..., None]
    return X, y

# -----------------------
# 4) 读取数据 & 预处理（先切分，再fit scaler）
# -----------------------
print("📂 正在读取并预处理数据...")
df = pd.read_csv(DATA_PATH)
df = df.select_dtypes(include=[np.number]).copy()

if TARGET_COL_NAME not in df.columns:
    raise ValueError(f"找不到目标列 {TARGET_COL_NAME}，当前列: {df.columns.tolist()}")

cols = [TARGET_COL_NAME] + [c for c in df.columns if c != TARGET_COL_NAME]
df = df[cols]

values = df.values.astype(np.float32)
N, D = values.shape
print(f"✅ 数据形状: {values.shape} (样本数={N}, 特征数={D})")
print(f"✅ 特征列: {df.columns.tolist()}")

# ✅ 三段切分：train / vali / test（严格按时间顺序）
train_end = int(N * TRAIN_RATIO)
vali_end = int(N * (TRAIN_RATIO + VALI_RATIO))

train_values = values[:train_end]
vali_values = values[train_end:vali_end]
test_values = values[vali_end:]

print(f"✅ 切分: train={len(train_values)}, vali={len(vali_values)}, test={len(test_values)}")

# X scaler：只 fit train
x_scaler = MinMaxScaler(feature_range=(0, 1))
x_scaler.fit(train_values)

train_x = x_scaler.transform(train_values)
vali_x = x_scaler.transform(vali_values)
test_x = x_scaler.transform(test_values)

# y scaler：只 fit train（STD）
y_scaler = StandardScaler()
y_scaler.fit(train_values[:, [0]])

train_y = y_scaler.transform(train_values[:, [0]])
vali_y = y_scaler.transform(vali_values[:, [0]])
test_y = y_scaler.transform(test_values[:, [0]])

# 在各自块内生成序列，避免跨边界泄漏
X_train, y_train = create_sequences(train_x, train_y, SEQ_LEN, PRED_LEN)
X_vali, y_vali = create_sequences(vali_x, vali_y, SEQ_LEN, PRED_LEN)
X_test, y_test = create_sequences(test_x, test_y, SEQ_LEN, PRED_LEN)

print(f"✅ Train: X={X_train.shape}, y={y_train.shape}")
print(f"✅ Vali : X={X_vali.shape}, y={y_vali.shape}")
print(f"✅ Test : X={X_test.shape}, y={y_test.shape}")

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=False
)
vali_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_vali), torch.from_numpy(y_vali)),
    batch_size=BATCH_SIZE, shuffle=False, drop_last=False
)
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
    batch_size=BATCH_SIZE, shuffle=False, drop_last=False
)

# -----------------------
# 5) 模型定义
# -----------------------
class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=96, num_layers=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)           # (B, D, seq_len)
        x = self.relu(self.conv1(x))     # (B, 64, seq_len)
        x = x.permute(0, 2, 1)           # (B, seq_len, 64)
        out, _ = self.lstm(x)            # (B, seq_len, hidden)
        last = out[:, -1, :]             # (B, hidden)
        pred = self.fc(last)             # (B, pred_len)
        return pred.unsqueeze(-1)        # (B, pred_len, 1)

model = CNN_LSTM(input_dim=D, hidden_dim=128, output_dim=PRED_LEN).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
early_stopping = EarlyStopping(save_path=model_save_path, patience=PATIENCE)

print(model)

# -----------------------
# 6) 训练
# -----------------------
print(f"\n🚀 开始训练: {setting}")

loss_train, loss_vali = [], []

for epoch in range(EPOCHS):
    t0 = time.time()
    model.train()
    train_losses = []

    for bx, by in train_loader:
        bx = bx.to(device)
        by = by.to(device)
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    vali_losses = []
    with torch.no_grad():
        for vx, vy in vali_loader:
            vx = vx.to(device)
            vy = vy.to(device)
            vp = model(vx)
            vali_losses.append(criterion(vp, vy).item())

    avg_train = float(np.mean(train_losses))
    avg_vali = float(np.mean(vali_losses))
    loss_train.append(avg_train)
    loss_vali.append(avg_vali)

    dt = time.time() - t0
    print(f"Epoch [{epoch+1:03d}/{EPOCHS}] | {dt:.2f}s | TrainLoss={avg_train:.6f} | ValiLoss={avg_vali:.6f}")

    early_stopping(avg_vali, model)
    if early_stopping.early_stop:
        print("⛔ 提前停止触发")
        break

np.save(os.path.join(results_dir, "loss_train.npy"), np.array(loss_train, dtype=np.float32))
np.save(os.path.join(results_dir, "loss_vali.npy"), np.array(loss_vali, dtype=np.float32))

# -----------------------
# 7) Test loss
# -----------------------
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

test_losses = []
with torch.no_grad():
    for tx, ty in test_loader:
        tx = tx.to(device)
        ty = ty.to(device)
        tp = model(tx)
        test_losses.append(criterion(tp, ty).item())

loss_test = float(np.mean(test_losses))
np.save(os.path.join(results_dir, "loss_test.npy"), np.array([loss_test], dtype=np.float32))

# -----------------------
# 8) 评估
# -----------------------
all_preds, all_trues = [], []
with torch.no_grad():
    for tx, ty in test_loader:
        tx = tx.to(device)
        tp = model(tx).cpu().numpy()   # (B, pred_len, 1)
        all_preds.append(tp)
        all_trues.append(ty.numpy())   # (B, pred_len, 1)

preds = np.concatenate(all_preds, axis=0)
trues = np.concatenate(all_trues, axis=0)

preds_s = preds.squeeze(-1)  # (N, pred_len)
trues_s = trues.squeeze(-1)  # (N, pred_len)

mae = mean_absolute_error(trues_s.flatten(), preds_s.flatten())
mse = mean_squared_error(trues_s.flatten(), preds_s.flatten())
rmse = float(np.sqrt(mse))
r2 = r2_score(trues_s.flatten(), preds_s.flatten())
mape = masked_mape(preds_s.flatten(), trues_s.flatten(), eps=1e-6)

print(f"✅ 结果 | MSE={mse:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f} | MAPE={mape:.2f}%")

np.save(os.path.join(results_dir, "pred.npy"), preds_s)
np.save(os.path.join(results_dir, "true.npy"), trues_s)
np.save(os.path.join(results_dir, "metrics.npy"), np.array([mae, mse, rmse, mape, r2], dtype=np.float32))

with open("result_long_term_forecast.txt", "a", encoding="utf-8") as f:
    f.write(setting + "\n")
    f.write(f"mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, r2:{r2:.4f}\n\n")

print(f"🌟 已保存至: {results_dir}")