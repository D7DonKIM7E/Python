import pandas as pd
import numpy as np
import sys

file_path = sys.argv[1]  # 'pred_and_target_pairs' file path
n_label = int(sys.argv[2])

df = pd.read_csv(file_path)
df = df.drop(df.columns[0], axis=1)
cmt = np.zeros((n_label, n_label), dtype=np.int)  # confusion matrix

for pair in df.values:
    pred, target = pair.tolist()
    cmt[target, pred] = cmt[target, pred] + 1

print(cmt)
