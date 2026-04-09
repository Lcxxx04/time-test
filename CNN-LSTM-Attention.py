import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# --- 1. 环境检测 ---
print("=" * 50)
print(f"🔍 当前运行环境: {sys.executable}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print("=" * 50)

# --- 2. 参数配置 ---
DATA_PATH = 'dataset/London/Block1_Final_1h.csv'
TARGET_COL_NAME = 'WHE'
SEQ_LEN = 512
PRED_LEN = 96
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-4
PATIENCE = 20
dataset_name = "LondonB1"

# ✅ 三段切分
TRAIN_RATIO = 0.7
VALI_RATIO = 0.1

setting = f"long_term_forecast_{dataset_name}_{SEQ_LEN}_{PRED_LEN}_CNN_LSTM_Attention_custom_ftMS_sl{SEQ_LEN}_pl{PRED_LEN}_Exp_0"
checkpoints_dir = os.path.join('./checkpoints', setting)
results_dir = os.path.join('./results', setting)
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
model_save_path = os.path.join(checkpoints_dir, 'checkpoint.pth')

# --- 3. 辅助功能 ---
def masked_mape_eps(preds, trues, eps=1e-6):
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
            print(f'EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(model)
            self.counter = 0

    def _save(self, model):
        torch.save(model.state_dict(), self.save_path)

def create_sequences(x_block, y_block, seq_len, pred_len):
    xs, ys = [], []
    T = len(x_block)
    for i in tqdm(range(T - seq_len - pred_len + 1), desc="生成序列"):
        xs.append(x_block[i:i + seq_len, :])
        ys.append(y_block[i + seq_len:i + seq_len + pred_len, 0])
    X = np.asarray(xs, dtype=np.float32)                 # (N, seq_len, D)
    y = np.asarray(ys, dtype=np.float32)[..., None]      # (N, pred_len, 1)
    return X, y

# --- 4. 数据处理（关键：先切分，再fit scaler）---
print("📂 正在预处理数据...")
df = pd.read_csv(DATA_PATH)
df = df.select_dtypes(include=[np.number]).copy()
if TARGET_COL_NAME not in df.columns:
    raise ValueError(f"找不到目标列 {TARGET_COL_NAME}，当前列: {df.columns.tolist()}")

cols = [TARGET_COL_NAME] + [c for c in df.columns if c != TARGET_COL_NAME]
df = df[cols]
values = df.values.astype(np.float32)
N, D = values.shape

train_end = int(N * TRAIN_RATIO)
vali_end = int(N * (TRAIN_RATIO + VALI_RATIO))

train_values = values[:train_end]
vali_values = values[train_end:vali_end]
test_values = values[vali_end:]

print(f"✅ 切分: train={len(train_values)}, vali={len(vali_values)}, test={len(test_values)}")

# X scaler：MinMax（fit train）
x_scaler = MinMaxScaler(feature_range=(0, 1))
x_scaler.fit(train_values)
train_x = x_scaler.transform(train_values)
vali_x = x_scaler.transform(vali_values)
test_x = x_scaler.transform(test_values)

# y scaler：StandardScaler（fit train，便于与 PatchTST 对齐）
y_scaler = StandardScaler()
y_scaler.fit(train_values[:, [0]])
train_y = y_scaler.transform(train_values[:, [0]])
vali_y = y_scaler.transform(vali_values[:, [0]])
test_y = y_scaler.transform(test_values[:, [0]])

X_train, y_train = create_sequences(train_x, train_y, SEQ_LEN, PRED_LEN)
X_vali, y_vali = create_sequences(vali_x, vali_y, SEQ_LEN, PRED_LEN)
X_test, y_test = create_sequences(test_x, test_y, SEQ_LEN, PRED_LEN)

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                          batch_size=BATCH_SIZE, shuffle=True)
vali_loader = DataLoader(TensorDataset(torch.from_numpy(X_vali), torch.from_numpy(y_vali)),
                         batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                         batch_size=BATCH_SIZE, shuffle=False)

# --- 5. 模型定义 ---
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: (B, T, H)
        attn_logits = self.attn(x)                    # (B, T, 1)
        attn_weights = torch.softmax(attn_logits, 1)  # (B, T, 1)
        context = torch.sum(x * attn_weights, dim=1)  # (B, H)
        return context

class CNN_LSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=96):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True, num_layers=2, dropout=0.1)
        self.attn = AttentionLayer(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.relu(self.bn1(self.conv1(x)))  # (B, 128, T)
        x = x.permute(0, 2, 1)  # (B, T, 128)

        lstm_out, _ = self.lstm(x)  # (B, T, H)
        lstm_out = lstm_out[:, -64:, :]  # ✅ 只对最近 64 步做 attention

        ctx = self.attn(lstm_out)  # ✅ (B, H) 只算一次
        out = self.fc(ctx)  # (B, P)
        return out.unsqueeze(-1)  # (B, P, 1)

model = CNN_LSTM_Attention(input_dim=D, output_dim=PRED_LEN).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
early_stopping = EarlyStopping(save_path=model_save_path, patience=PATIENCE)

# --- 6. 训练循环 ---
print(f"🚀 开始训练: {setting}")
loss_train, loss_vali = [], []

for epoch in range(EPOCHS):
    t0 = time.time()
    model.train()
    tl = []

    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()
        tl.append(loss.item())

    model.eval()
    vl = []
    with torch.no_grad():
        for vx, vy in vali_loader:
            vx, vy = vx.to(device), vy.to(device)
            vl.append(criterion(model(vx), vy).item())

    avg_t = float(np.mean(tl))
    avg_v = float(np.mean(vl))
    loss_train.append(avg_t)
    loss_vali.append(avg_v)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | {time.time()-t0:.2f}s | Train Loss: {avg_t:.6f} | Vali Loss: {avg_v:.6f}")

    scheduler.step(avg_v)
    early_stopping(avg_v, model)
    if early_stopping.early_stop:
        print("⛔ 提前停止触发")
        break

np.save(os.path.join(results_dir, "loss_train.npy"), np.array(loss_train, dtype=np.float32))
np.save(os.path.join(results_dir, "loss_vali.npy"), np.array(loss_vali, dtype=np.float32))

# --- 7. Test loss（best checkpoint）---
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

test_losses = []
with torch.no_grad():
    for tx, ty in test_loader:
        tx, ty = tx.to(device), ty.to(device)
        test_losses.append(criterion(model(tx), ty).item())

loss_test = float(np.mean(test_losses))
np.save(os.path.join(results_dir, "loss_test.npy"), np.array([loss_test], dtype=np.float32))

# --- 8. 输出 pred/true/metrics（用 test 集）---
all_preds, all_trues = [], []
with torch.no_grad():
    for tx, ty in test_loader:
        tx = tx.to(device)
        all_preds.append(model(tx).cpu().numpy())  # (B,P,1)
        all_trues.append(ty.numpy())               # (B,P,1)

preds = np.concatenate(all_preds, axis=0)
trues = np.concatenate(all_trues, axis=0)

preds_s = preds.squeeze(-1)  # (N,P)
trues_s = trues.squeeze(-1)  # (N,P)

mae = mean_absolute_error(trues_s.flatten(), preds_s.flatten())
mse = mean_squared_error(trues_s.flatten(), preds_s.flatten())
rmse = float(np.sqrt(mse))
r2 = r2_score(trues_s.flatten(), preds_s.flatten())
mape = masked_mape_eps(preds_s.flatten(), trues_s.flatten(), eps=1e-6)

print(f'✅ 结果 | MSE={mse:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f} | MAPE={mape:.2f}%')

np.save(os.path.join(results_dir, 'pred.npy'), preds_s)
np.save(os.path.join(results_dir, 'true.npy'), trues_s)
np.save(os.path.join(results_dir, 'metrics.npy'), np.array([mae, mse, rmse, mape, r2], dtype=np.float32))

with open("result_long_term_forecast.txt", 'a', encoding='utf-8') as f:
    f.write(setting + "\n")
    f.write(f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, r2:{r2:.4f}\n\n')

print(f"🌟 实验成功！结果已存至 {results_dir}")