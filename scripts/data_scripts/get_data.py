#!/usr/bin/python3

import pandas as pd

# скачиваем данные и сохраняем в папке raw 

train = pd.read_csv('https://drive.google.com/uc?id=1_wQ8mbJL7C0X9RT89lmW26gaVqk7XyTd')

train.to_csv("../../data/raw/train.csv", index=False)
