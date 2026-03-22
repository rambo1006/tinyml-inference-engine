import numpy as np


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def forward(x, W1, b1, W2, b2):
    h = relu(W1 @ x + b1)
    return softmax(W2 @ h + b2)


def quantize(arr):
    if arr.max() == arr.min():
        return np.zeros_like(arr, dtype=np.int8), 1.0, 0

    scale      = (arr.max() - arr.min()) / 255.0
    zero_point = int(np.clip(round(-128 - arr.min() / scale), -128, 127))
    q          = np.clip(np.round(arr / scale + zero_point), -128, 127).astype(np.int8)
    return q, scale, zero_point


def dequantize(q, scale, zero_point):
    return scale * (q.astype(np.float32) - zero_point)


def quantized_layer(x_q, W_q, bias, x_scale, x_zp, W_scale, W_zp):
    acc = (W_q.astype(np.int32) - W_zp) @ (x_q.astype(np.int32) - x_zp)
    return acc.astype(np.float32) * x_scale * W_scale + bias


W1 = np.array([[ 1.0, -1.0],
               [-1.0,  1.0]], dtype=np.float32)
b1 = np.array([0.0, 0.0], dtype=np.float32)
W2 = np.array([[-1.0, -1.0],
               [ 1.0,  1.0]], dtype=np.float32)
b2 = np.array([0.5, -0.5], dtype=np.float32)

inputs  = [(0,0), (0,1), (1,0), (1,1)]
targets = [0, 1, 1, 0]

print("Float32 inference")
print(f"{'Input':<10} {'Target':<10} {'Predicted':<10} {'OK'}")
print("-" * 38)
float_preds = []
for inp, target in zip(inputs, targets):
    x    = np.array(inp, dtype=np.float32)
    pred = int(np.argmax(forward(x, W1, b1, W2, b2)))
    float_preds.append(pred)
    print(f"{str(inp):<10} {target:<10} {pred:<10} {'YES' if pred==target else 'NO'}")

W1_q, W1_s, W1_zp = quantize(W1)
W2_q, W2_s, W2_zp = quantize(W2)

print("\nQuantized int8 inference")
print(f"{'Input':<10} {'Float32':<10} {'Int8':<10} {'Match'}")
print("-" * 38)
for inp, fp in zip(inputs, float_preds):
    x_f = np.array(inp, dtype=np.float32)
    x_q, x_s, x_zp = quantize(x_f)

    h_raw = quantized_layer(x_q, W1_q, b1, x_s, x_zp, W1_s, W1_zp)
    h_q, h_s, h_zp = quantize(relu(h_raw))

    out = softmax(quantized_layer(h_q, W2_q, b2, h_s, h_zp, W2_s, W2_zp))
    ip  = int(np.argmax(out))
    print(f"{str(inp):<10} {fp:<10} {ip:<10} {'YES' if ip==fp else 'NO'}")

print("\nMemory")
f32_bytes = W1.nbytes + W2.nbytes
int8_bytes = W1_q.nbytes + W2_q.nbytes
print(f"float32 : {f32_bytes} bytes")
print(f"int8    : {int8_bytes} bytes  ({f32_bytes//int8_bytes}x smaller)")