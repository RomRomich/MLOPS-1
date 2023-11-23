import sys
import os
import pandas as pd

# Преобразование текстовых признаков в числовые, смена типов,
# Сохранение файлов в stage3

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 change_text_to_numeric.py data-file\n")
    sys.exit(1)

f_input = sys.argv[1]
os.makedirs(os.path.join("data", "stage3"), exist_ok=True)

# Датасет для обработки
df = pd.read_csv(f_input)

# Удаление признаков, которые не оказывают значитального влияния на модель обучения
df.drop([
"YrSold", "MSSubClass", "LotConfig", "RoofStyle", "HouseStyle", "LotShape",
"Exterior1st", "Exterior2nd", "BsmtFullBath", "BedroomAbvGr", "HalfBath",
"BsmtFinType2", "MoSold", "YrSold"
],
axis=1, inplace=True)

# Замена текстовых признаков на числовые
df["SaleCondition"] = pd.factorize(df["SaleCondition"])[0]
df["SaleType"] = pd.factorize(df["SaleType"])[0]
df["Condition1"] = pd.factorize(df["Condition1"])[0]


df.to_csv("data/stage3/train.csv", index=False)
