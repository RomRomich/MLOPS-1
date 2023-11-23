import os
import sys
import pickle
import json
import pandas as pd

# Получение метрик работ модели на тестовые данные с помощью этого файла

# Тестовый файл и обученная модель
if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py data-file model\n")
    sys.exit(1)

# Загрузка тренировочного датасета и разделение признаков и меток
# df = pd.read_csv(sys.argv[1], header=None)
df = pd.read_csv(sys.argv[1])

# Разделение признаков и меток
X_test = df.iloc[:,0:-1]
y_test = df.iloc[:,-1].astype('int')

# Загрузка модели
with open(sys.argv[2], "rb") as fd:
    clf = pickle.load(fd)

# Получение метрик работы модели на тестовых данных
score = clf.score(X_test, y_test)

prc_file = os.path.join("evaluate", "score.json")
os.makedirs(os.path.join("evaluate"), exist_ok=True)

with open(prc_file, "w") as fd:
    json.dump({"score": score}, fd)
