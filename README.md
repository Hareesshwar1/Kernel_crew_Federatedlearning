![image](https://github.com/user-attachments/assets/37f65c65-9820-4e5a-8bad-62b23b761668)

-------------------------------------------------------------------------------------------------------------
|This script helps visualize how data is spread across 12 different clients in a federated learning setup using the CIFAR-10 dataset.|

The dataset is split using a Dirichlet distribution with a parameter of 0.05, which creates an intentionally uneven distribution to mimic real-world non-IID data scenarios.

For each client, the script loads training data from a file named train_data.pth, which contains image-label pairs.

It extracts all the labels and counts how many samples belong to each class using Python’s built-in tools.

A bar chart is then created for each client, showing the number of samples available for each of the ten CIFAR-10 classes.

These charts make it easy to see how imbalanced the label distribution is for each client.

This is especially useful in federated learning experiments where unequal data can affect the training performance of individual clients.

By visualizing the distribution, we can better understand how client performance might vary and make more informed decisions about model aggregation and personalization strategies.


-------------------------------------------------------------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////////////////////////////////
-------------------------------------------------------------------------------------------------------------


![image](https://github.com/user-attachments/assets/0bc14d2c-9cf8-4db2-b9bb-d95879e1113d)


![image](https://github.com/user-attachments/assets/e017b15e-9ac7-49f7-b4b1-d0e4150dd284)


![2](https://github.com/user-attachments/assets/c30a06f9-afe3-40dc-94cc-5b41445072a0)


![3](https://github.com/user-attachments/assets/c24db81f-c5a1-4cef-aeff-a9dde1b71722)


![4](https://github.com/user-attachments/assets/5bcdb45c-cfe5-4a6a-a663-159f1d196fa0)


![5](https://github.com/user-attachments/assets/1d1cd5df-3666-4143-9d08-ab51e3abba33)


![6](https://github.com/user-attachments/assets/6921aa7c-be5b-4e18-afea-96fa19208271)


![7](https://github.com/user-attachments/assets/c820bfc2-5e44-4847-8b55-b30253e1250b)


![8](https://github.com/user-attachments/assets/104dac05-63fb-45b8-b88c-52c7c93b3ebb)


![9](https://github.com/user-attachments/assets/d2de744d-e3e4-4caf-9b0d-4a11c8067ad4)


![10](https://github.com/user-attachments/assets/d0a8541f-2012-40c5-af15-d2621f9f50f2)


![11](https://github.com/user-attachments/assets/a78f071c-a499-498a-a802-4b30d188c23a)


![12](https://github.com/user-attachments/assets/dd57c7d1-2305-4d0d-82d1-f32ef0c3ef52)


-------------------------------------------------------------------------------------------------------------
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
-------------------------------------------------------------------------------------------------------------

## Number of Samples per Client
![image](https://github.com/Hareesshwar1/Kernel_crew_Federatedlearning/blob/main/images/NumberOfSamplesPerClient.jpeg)

## Label Distribution per Client (%)
![image](https://github.com/Hareesshwar1/Kernel_crew_Federatedlearning/blob/main/images/ClientLevelDistributionPerClient.jpeg)


## Training Time and Memory Usage vs Batch Size
![image](https://github.com/Hareesshwar1/Kernel_crew_Federatedlearning/blob/main/images/TT_MUvsBS.jpeg)
