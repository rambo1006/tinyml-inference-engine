import numpy as np
import os

X_val = np.load("data/X_val.npy")
y_val = np.load("data/y_val.npy")

n_samples, n_features = X_val.shape
print(f"Exporting {n_samples} validation samples, {n_features} features each")


def float_matrix_to_c(name, arr):
    rows = []
    for row in arr:
        vals = ", ".join(f"{v:.8f}f" for v in row)
        rows.append(f"  {{{vals}}}")
    body = ",\n".join(rows)
    return f"const float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n{body}\n}};\n"


def int_array_to_c(name, arr):
    vals = ", ".join(str(int(v)) for v in arr)
    return f"const int {name}[{len(arr)}] = {{{vals}}};\n"


os.makedirs("weights", exist_ok=True)
with open("weights/validation_data.h", "w") as f:
    f.write("#ifndef VALIDATION_DATA_H\n#define VALIDATION_DATA_H\n\n")
    f.write(f"#define NUM_VAL_SAMPLES {n_samples}\n\n")
    f.write("/* Raw (un-normalized) MFCC features -- normalize_input() in the\n")
    f.write("   C engine applies NORM_MEAN/NORM_STD before quantizing, exactly\n")
    f.write("   like X_val = (X_val - mean) / std in week4_train.py. */\n")
    f.write(float_matrix_to_c("VAL_FEATURES", X_val))
    f.write("\n")
    f.write(int_array_to_c("VAL_LABELS", y_val))
    f.write("\n#endif\n")

print("weights/validation_data.h exported successfully.")
print(f"File size check: {os.path.getsize('weights/validation_data.h') / 1024:.1f} KB")