# Handling Class Imbalance in Continual Learning based Network Intrusion Detection System
In this work, we try handling infamous **class imbalance problem** frequently seen in intrusion detection datasets. Specifically, we study this problem under the application of **continual learning (CL)** to the intrusion detection. Under CL paradigm, the learning model will be more flexible to adapt to the newly seen attack pattern with minimal overhead.

# Proposed System Model

![ScreenShot](/figure/arch.png)

# Datasets
CICIDS 2017 - https://www.unb.ca/cic/datasets/ids-2017.html <br>
<br>Dataset contains 8 csv files, input to the datapreprocessing code. They are
- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
- Friday-WorkingHours-Morning.pcap_ISCX.csv
- Monday-WorkingHours.pcap_ISCX.csv
- Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
- Tuesday-WorkingHours.pcap_ISCX.csv
- Wednesday-workingHours.pcap_ISCX.csv

# Different **Task Orders**
In this work we formulated five different task orders. They are
- Benign in task1 task_order
- Benign in task2 task_order
- Benign in task3 task_order
- Benign in task4 task_order
- Benign in task5 task_order

To execute different task order, assign the variable **task_order** to the one of the above task order.




# Software setup details
 - Ubuntu >= 18.04
 - Python >= 3.8 (https://www.python.org/downloads/)
 - Anaconda (https://docs.anaconda.com/anaconda/install/linux/) 
 - Pytorch (https://pytorch.org/), latest and stable build say 1.10, prefered to install with conda
 - Avalanche continual learning library (Prefered way to install - https://avalanche.continualai.org/getting-started/how-to-install#install-avalanche-with-anaconda)

# Code execution

To run the code smoothly, follow the below steps in the same order

- Ensure the correct software setup installation
- Download and place all the csv files in the **code** directory
- Run preprocess_dataset_ids17.py
- Now you can run any of mlp or cnn architecture based code files
- To run for different task order, follow steps in **Different Task Orders** section


# Miscellaneous

- Avalanche documentation - https://avalanche-api.continualai.org/en/latest/
- Avalanche youtube channel - https://www.youtube.com/channel/UCD9_bqN3gX-TLxcr47vvMmA
- Getting start with pytorch - https://pytorch.org/tutorials/



# Citation
@inbook{10.1145/3486001.3486231,<br>
author = {Amalapuram, Suresh Kumar and Reddy, Thushara Tippi and Channappayya, Sumohana S. and Tamma, Bheemarjuna Reddy},<br>
title = {On Handling Class Imbalance in Continual Learning Based Network Intrusion Detection Systems},<br>
year = {2021},<br>
publisher = {Association for Computing Machinery},<br>
address = {New York, NY, USA},<br>
url = {https://doi.org/10.1145/3486001.3486231},<br>
booktitle = {The First International Conference on AI-ML-Systems},<br>
articleno = {11},<br>
numpages = {7}<br>
}
