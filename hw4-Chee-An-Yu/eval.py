import pandas as pd
import numpy as np
file1 = "/home/robot/Desktop/p1_valid.txt"
file2 = "/home/robot/hw4-Chee-An-Yu/hw4_data/TrimmedVideos/label/gt_valid.csv"
predict = pd.read_csv(file1,header=None,squeeze=True)
predict = predict.tolist()

truth = pd.read_csv(file2)
# truth = truth.sort_values(['Video_name']).reset_index(drop=True)
truth = truth["Action_labels"].tolist()
print("predict",predict)
print("truth",truth)

correct = (np.array(predict) == np.array(truth)).sum()
total = len(predict)
accuracy = correct / total
print(accuracy)
