## ProcessMiner QCRE Data Challenge Competition

This document provides instructions on how to run the model for the **ProcessMiner QCRE Data Challenge Competition**.

---

# **A Knowledge guided ML Framework for Predicting Fungal Spore Concentration**

### Md Asif Bin Syed, Azmine Toushik Wasi, and Imtiaz Ahmed

---
---

# Dataset Placement

Both the `trainset.xlsx` and `testset.xlsx` files should be placed in the same directory where there `predictions.py` file is placed.

To use data from any other locations, please change the `xlsx_path_train_data` and `xlsx_path_test_data` variables *(can be located at the very begining of both files)* accordingly with respective file locations.

Current file locations are denoted here:

```
xlsx_path_train_data = "trainset.xlsx"
xlsx_path_test_data = "testset.xlsx"
```

---
# Running the Model 

## Running the Model using a .py file

To run the model using the `predictions.py` file, you need to install all the required modules listed in the `requirements.txt` file. You can install these modules by running the following command:

```
pip install -r requirements.txt
```

Once you have installed the necessary modules, you can run the model by executing the following command:

```
python predictions.py
```
---

## Running the Model using an .ipynb file (Jupyter Notebook)

To run the model using the `train.ipynb` Jupyter Notebook file, you first need to install all the required modules listed in the `requirements_ipynb.txt` file. You can install these modules by running the following command:

```
pip install -r requirements_ipynb.txt
```

After installing the modules, you can run the model by opening the `predictions.ipynb` file in Jupyter Notebook and clicking on the "Run All" button or manually running each code cell sequentially.

---
# Model Architecture

![](figures\architecture.jpg)