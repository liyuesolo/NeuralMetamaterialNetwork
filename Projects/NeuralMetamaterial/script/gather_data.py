import os
import numpy as np

full_data = []


for i in range(400):
    for line in open(str(i)+"/data.txt").readlines():
        full_data.append(line)

indices = [i for i in range(len(full_data))]
shuffle = True
suffix = ""
if shuffle:
    np.random.shuffle(indices)
    suffix = "_shuffled"
full_data = np.array(full_data)
full_data = full_data[indices]

f = open("all_data_IH30"+suffix+".txt", "w+")
for line in full_data:
    f.write(line)
f.close()
