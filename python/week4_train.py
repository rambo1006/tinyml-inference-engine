import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load and Normalize Data
# ---------------------------------------------------------
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_val   = np.load("data/X_val.npy")
y_val   = np.load("data/y_val.npy")

# Compute statistics only on the training set
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0) + 1e-8

# Apply normalization
X_train = (X_train - mean) / std
X_val   = (X_val - mean) / std

# Save for the C engine
os.makedirs("data", exist_ok=True)
np.save("data/norm_mean.npy", mean)
np.save("data/norm_std.npy", std)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# ---------------------------------------------------------
# 2. Define the Model
# ---------------------------------------------------------
class KeywordMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(13, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 3)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        
        x = self.layer2(x)
        x = self.relu(x)
        
        x = self.layer3(x)
        return x

model = KeywordMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


# ---------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------
train_losses = []
val_accuracies = []

print("Training...")
print("Epoch | Loss       | Val Accuracy")
print("----------------------------------")

for epoch in range(30):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        preds = val_outputs.argmax(dim=1)
        
        correct_preds = (preds == y_val_t).float().sum()
        val_acc = (correct_preds / len(y_val_t)).item() * 100

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    val_accuracies.append(val_acc)

    if (epoch + 1) % 5 == 0:
        print(f"{epoch + 1:<5} | {avg_loss:<10.4f} | {val_acc:.1f}%")

best_acc = max(val_accuracies)
print(f"\nBest validation accuracy: {best_acc:.1f}%")


# ---------------------------------------------------------
# 4. Quantization Functions
# ---------------------------------------------------------
def quantize_array(arr, n_bits=8):
    q_min = -(2 ** (n_bits - 1))
    q_max = (2 ** (n_bits - 1)) - 1
    
    if arr.max() == arr.min():
        return np.zeros_like(arr, dtype=np.int8), 1.0, 0
        
    scale = (arr.max() - arr.min()) / (q_max - q_min)
    
    zero_point = q_min - (arr.min() / scale)
    zero_point = int(np.clip(round(zero_point), q_min, q_max))
    
    q_arr = np.round((arr / scale) + zero_point)
    q_arr = np.clip(q_arr, q_min, q_max).astype(np.int8)
    
    return q_arr, scale, zero_point


# ---------------------------------------------------------
# 5. Extract and Quantize Weights
# ---------------------------------------------------------
model.eval()

W1 = model.layer1.weight.detach().numpy()
b1 = model.layer1.bias.detach().numpy()
W2 = model.layer2.weight.detach().numpy()
b2 = model.layer2.bias.detach().numpy()
W3 = model.layer3.weight.detach().numpy()
b3 = model.layer3.bias.detach().numpy()

W1_q, W1_s, W1_zp = quantize_array(W1)
W2_q, W2_s, W2_zp = quantize_array(W2)
W3_q, W3_s, W3_zp = quantize_array(W3)


# ---------------------------------------------------------
# 6. Int8 Inference Validation
# ---------------------------------------------------------
print("\nRunning int8 inference on validation set...")
correct = 0

for i in range(len(X_val)):
    x_f = X_val[i]
    x_q, x_s, x_zp = quantize_array(x_f)

    # Layer 1
    h1_acc = (W1_q.astype(np.int32) - W1_zp) @ (x_q.astype(np.int32) - x_zp)
    h1_f = h1_acc.astype(np.float32) * x_s * W1_s + b1
    h1 = np.maximum(0, h1_f)
    h1_q, h1_s, h1_zp = quantize_array(h1)

    # Layer 2
    h2_acc = (W2_q.astype(np.int32) - W2_zp) @ (h1_q.astype(np.int32) - h1_zp)
    h2_f = h2_acc.astype(np.float32) * h1_s * W2_s + b2
    h2 = np.maximum(0, h2_f)
    h2_q, h2_s, h2_zp = quantize_array(h2)

    # Layer 3
    out_acc = (W3_q.astype(np.int32) - W3_zp) @ (h2_q.astype(np.int32) - h2_zp)
    out = out_acc.astype(np.float32) * h2_s * W3_s + b3

    if np.argmax(out) == y_val[i]:
        correct += 1

int8_acc = (correct / len(X_val)) * 100

print(f"float32 accuracy : {best_acc:.1f}%")
print(f"int8    accuracy : {int8_acc:.1f}%")
print(f"Accuracy drop    : {best_acc - int8_acc:.1f}%")


# ---------------------------------------------------------
# 7. Export to C Header
# ---------------------------------------------------------
def array_to_c(name, arr, ctype="int8_t"):
    flat_arr = arr.flatten()
    vals = []
    for v in flat_arr:
        vals.append(str(int(v)))
    
    vals_str = ", ".join(vals)
    return f"// shape: {arr.shape}\nconst {ctype} {name}[] = {{{vals_str}}};\n"

def float_array_to_c(name, arr):
    flat_arr = arr.flatten()
    vals = []
    for v in flat_arr:
        vals.append(str(v))
        
    vals_str = ", ".join(vals)
    return f"const float {name}[] = {{{vals_str}}};\n"


os.makedirs("weights", exist_ok=True)

with open("weights/weights.h", "w") as f:
    f.write("#ifndef WEIGHTS_H\n")
    f.write("#define WEIGHTS_H\n\n")
    f.write("#include <stdint.h>\n\n")
    
    f.write("#define INPUT_SIZE 13\n")
    f.write("#define HIDDEN1_SIZE 64\n")
    f.write("#define HIDDEN2_SIZE 32\n")
    f.write("#define OUTPUT_SIZE 3\n\n")
    
    f.write(f"const float SCALE_W1 = {W1_s};\n")
    f.write(f"const int ZP_W1 = {W1_zp};\n")
    f.write(f"const float SCALE_W2 = {W2_s};\n")
    f.write(f"const int ZP_W2 = {W2_zp};\n")
    f.write(f"const float SCALE_W3 = {W3_s};\n")
    f.write(f"const int ZP_W3 = {W3_zp};\n\n")
    
    f.write(array_to_c("W1_data", W1_q))
    f.write(float_array_to_c("b1_data", b1))
    
    f.write(array_to_c("W2_data", W2_q))
    f.write(float_array_to_c("b2_data", b2))
    
    f.write(array_to_c("W3_data", W3_q))
    f.write(float_array_to_c("b3_data", b3))
    
    f.write(float_array_to_c("NORM_MEAN", mean))
    f.write(float_array_to_c("NORM_STD", std))
    
    f.write("\n#endif\n")

print("\nweights/weights.h exported successfully.")


# ---------------------------------------------------------
# 8. Memory Benchmark and Plotting
# ---------------------------------------------------------
f32_bytes = W1.nbytes + W2.nbytes + W3.nbytes
int8_bytes = W1_q.nbytes + W2_q.nbytes + W3_q.nbytes
total_budget = 262144  # 256 KB

print("\nMemory")
print(f"  float32 weights : {f32_bytes} bytes ({f32_bytes/1024:.1f} KB)")
print(f"  int8    weights : {int8_bytes} bytes ({int8_bytes/1024:.1f} KB)")
print(f"  Reduction       : {f32_bytes/int8_bytes:.1f}x")
print(f"  Total budget    : {total_budget} bytes (256 KB)")
print(f"  Used            : {int8_bytes} bytes ({int8_bytes/1024:.2f} KB)")
print(f"  Remaining       : {total_budget - int8_bytes} bytes ({(total_budget - int8_bytes)/1024:.1f} KB)")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Training results — TinyML keyword spotting", fontsize=13)

ax1.plot(train_losses, color="#534AB7")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training loss")
ax1.grid(True, alpha=0.3)

ax2.plot(val_accuracies, color="#1D9E75")
ax2.axhline(y=best_acc, color="#D85A30", linestyle="--", alpha=0.7, label=f"Best: {best_acc:.1f}%")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Validation accuracy")
ax2.legend()
ax2.grid(True, alpha=0.3)

os.makedirs("results", exist_ok=True)
plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nChart saved to results/training_curves.png")
print("Update results/benchmark.md with your numbers.")
print("Next: plug weights.h into inference_engine.c")