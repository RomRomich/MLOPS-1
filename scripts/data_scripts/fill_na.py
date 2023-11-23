import sys
import os
import pandas as pd

# Работа с пропущенными значениями, сохранение файла в stage1

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 fill_na.py data-file\n")
    sys.exit(1)

f_input = sys.argv[1]
os.makedirs(os.path.join("data", "stage1"), exist_ok=True)

# Датасет для обработки
df = pd.read_csv(f_input)

# Удаление признаков для которых много пропусков
_to_remove = df.columns[df.count() < 800]
df.drop(_to_remove, axis=1, inplace=True)

NUM_COLS = list(df.select_dtypes(include='number').columns)
CAT_COLS = list(df.select_dtypes(exclude='number').columns)

df[NUM_COLS] = df[NUM_COLS].fillna(0) 
df[CAT_COLS] = df[CAT_COLS].fillna("") 

df.to_csv("data/stage1/train.csv", index=False)
