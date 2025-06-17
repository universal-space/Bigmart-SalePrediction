# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    train = pd.read_csv('data/Train.csv')
    test = pd.read_csv('data/Test.csv')
    return train, test

def perform_eda(train):
    print(train.head())
    print(train.info())
    print(train.describe())
    sns.histplot(train['Item_Outlet_Sales'])
    plt.title("Sales Distribution")
    plt.show()
