# -*- coding: utf-8 -*-
"""kernelcrew_task1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eDUaUiT_R9BlaGR6m_6YEX6NT-T_0VFU
"""

!wget "https://www.dropbox.com/scl/fi/3bqhiz8uzcol2ge28ruce/data.zip?rlkey=ljhilzs8qam2m0mohru3otmx7&st=wkc7ehq7&dl=0" -O data.zip

!unzip data.zip

import os
import pickle
import matplotlib.pyplot as plt
base_path = r"CIFAR10_dirichlet0.05_12"
dataset_id = "CIFAR10_dirichlet0.05_12"

for cid in range(12):

    # Correctly build the full path
    dataset_path = os.path.join(base_path, f"part_{cid}", dataset_id)
    print("reading file from",dataset_path)
    try:
        file_path = os.path.join(dataset_path, "train_data.pth")
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        print(f"Exception caught from CNN dataloader while loading {file_path} :: {e}")
    from collections import Counter
    # Extract labels from the dataset
    labels = [label for _, label in dataset]
    label_counts = Counter(labels)
    print(f"number of samples for Client#{cid}",len(dataset))

    labels_sorted = sorted(label_counts.keys())
    counts = [label_counts[label] for label in labels_sorted]

    plt.figure(figsize=(8, 5))
    plt.bar(labels_sorted, counts, color='skyblue')
    plt.xlabel("Class Labels")
    plt.ylabel("Number of Samples")
    plt.title(f"Label Distribution for Client #{cid}")
    plt.xticks(labels_sorted)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

