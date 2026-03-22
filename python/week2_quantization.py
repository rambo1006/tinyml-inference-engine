import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def forward(x, W1, b1, W2, b2):
    h = relu(W1 @ x + b1)
    return softmax(W2 @ h + b2)


def quantize(arr, n_bits):
    if arr.max() == arr.min():
        return np.zeros_like(arr, dtype=np.int8), 1.0, 0

    q_min = -(2 ** (n_bits - 1))
    q_max =  (2 ** (n_bits - 1)) - 1

    scale      = (arr.max() - arr.min()) / (q_max - q_min)
    zero_point = int(np.clip(round(q_min - arr.min() / scale), q_min, q_max))
    q          = np.clip(np.round(arr / scale + zero_point), q_min, q_max).astype(np.int8)
    return q, scale, zero_point


def dequantize(q, scale, zero_point):
    return scale * (q.astype(np.float32) - zero_point)


def reconstruction_error(original, q, scale, zero_point):
    recovered = dequantize(q, scale, zero_point)
    return float(np.abs(original - recovered).mean())


def quantized_layer(x_q, W_q, bias, x_scale, x_zp, W_scale, W_zp):
    acc = (W_q.astype(np.int32) - W_zp) @ (x_q.astype(np.int32) - x_zp)
    return acc.astype(np.float32) * x_scale * W_scale + bias


W1 = np.array([[ 1.0, -1.0],
               [-1.0,  1.0]], dtype=np.float32)
b1 = np.array([0.0, 0.0],    dtype=np.float32)
W2 = np.array([[-1.0, -1.0],
               [ 1.0,  1.0]], dtype=np.float32)
b2 = np.array([0.5, -0.5],   dtype=np.float32)

inputs  = [(0,0), (0,1), (1,0), (1,1)]
targets = [0, 1, 1, 0]

float_preds = []
for inp in inputs:
    x = np.array(inp, dtype=np.float32)
    float_preds.append(int(np.argmax(forward(x, W1, b1, W2, b2))))


bit_depths  = [8, 7, 6, 5, 4, 3, 2]
w1_errors   = []
w2_errors   = []
correct_at  = []

print(f"{'Bits':<6} {'W1 error':<14} {'W2 error':<14} {'Correct/4'}")
print("-" * 45)

for bits in bit_depths:
    W1_q, W1_s, W1_zp = quantize(W1, bits)
    W2_q, W2_s, W2_zp = quantize(W2, bits)

    e1 = reconstruction_error(W1, W1_q, W1_s, W1_zp)
    e2 = reconstruction_error(W2, W2_q, W2_s, W2_zp)
    w1_errors.append(e1)
    w2_errors.append(e2)

    correct = 0
    for inp, fp in zip(inputs, float_preds):
        x_f = np.array(inp, dtype=np.float32)
        x_q, x_s, x_zp = quantize(x_f, bits)

        h_raw = quantized_layer(x_q, W1_q, b1, x_s, x_zp, W1_s, W1_zp)
        h_relu = relu(h_raw)
        h_q, h_s, h_zp = quantize(h_relu, bits)

        out = softmax(quantized_layer(h_q, W2_q, b2, h_s, h_zp, W2_s, W2_zp))
        if int(np.argmax(out)) == fp:
            correct += 1

    correct_at.append(correct)
    print(f"{bits:<6} {e1:<14.6f} {e2:<14.6f} {correct}/4")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Quantization analysis — TinyML inference engine", fontsize=13)

ax1.plot(bit_depths, w1_errors, marker='o', label='W1 error', color='#534AB7')
ax1.plot(bit_depths, w2_errors, marker='s', label='W2 error', color='#1D9E75')
ax1.set_xlabel("Bit depth")
ax1.set_ylabel("Mean reconstruction error")
ax1.set_title("Weight reconstruction error vs bit depth")
ax1.legend()
ax1.invert_xaxis()
ax1.grid(True, alpha=0.3)
ax1.axvline(x=8, color='gray', linestyle='--', alpha=0.5, label='int8 target')

ax2.bar([str(b) for b in bit_depths], correct_at, color='#378ADD', alpha=0.8)
ax2.set_xlabel("Bit depth")
ax2.set_ylabel("Correct predictions (out of 4)")
ax2.set_title("Prediction accuracy vs bit depth")
ax2.set_ylim(0, 5)
ax2.axhline(y=4, color='#1D9E75', linestyle='--', alpha=0.7, label='Perfect (4/4)')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("/home/claude/results/quantization_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved to /home/claude/results/quantization_analysis.png")


print("\nPer-layer scale factors (what your C engine will use):")
for name, W in [("W1", W1), ("W2", W2)]:
    _, s, zp = quantize(W, 8)
    print(f"  {name}: scale={s:.6f}  zero_point={zp}")
