This script helps visualize how data is spread across 12 different clients in a federated learning setup using the CIFAR-10 dataset.

The dataset is split using a Dirichlet distribution with a parameter of 0.05, which creates an intentionally uneven distribution to mimic real-world non-IID data scenarios.

For each client, the script loads training data from a file named train_data.pth, which contains image-label pairs.

It extracts all the labels and counts how many samples belong to each class using Python’s built-in tools.

A bar chart is then created for each client, showing the number of samples available for each of the ten CIFAR-10 classes.

These charts make it easy to see how imbalanced the label distribution is for each client.

This is especially useful in federated learning experiments where unequal data can affect the training performance of individual clients.

By visualizing the distribution, we can better understand how client performance might vary and make more informed decisions about model aggregation and personalization strategies.






![image](https://github.com/user-attachments/assets/0bc14d2c-9cf8-4db2-b9bb-d95879e1113d)

![image](https://github.com/user-attachments/assets/1a3800ac-1d40-4222-9a88-a4dedca846d9)

## Number of Samples per Client
![image](https://github.com/Hareesshwar1/Kernel_crew_Federatedlearning/blob/main/images/NumberOfSamplesPerClient.jpeg)

## Label Distribution per Client (%)
![image](https://github.com/Hareesshwar1/Kernel_crew_Federatedlearning/blob/main/images/ClientLevelDistributionPerClient.jpeg)


## Training Time and Memory Usage vs Batch Size
![image](https://github.com/Hareesshwar1/Kernel_crew_Federatedlearning/blob/main/images/TT_MUvsBS.jpeg)
