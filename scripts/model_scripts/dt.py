import sys
import os
import yaml
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis


# Обучение и сохранение модели в *.pkl


if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython dt.py data-file model \n")
    sys.exit(1)

f_input = sys.argv[1]
f_output = os.path.join("models", sys.argv[2])
os.makedirs(os.path.join("models"), exist_ok=True)

CLASSIFICATION_TARGET = "OverallQual"
params = yaml.safe_load(open("params.yaml"))["train"]
p_n_neighbors = params["n_neighbors"]
p_weights = params["weights"]

# Загрузка тренировочного датасета и разделение признаков и меток
df = pd.read_csv(f_input)

# Отделение признаков и меток
X_train = df.iloc[:,0:-1]
y_train = df.iloc[:,-1].astype('int')

# Пайп финальной подготовки и обучения датасета
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()), # StandardScaler or MinMaxScaler
    ('NCA', NeighborhoodComponentsAnalysis(n_components=8))
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False))
])

preprocessors = ColumnTransformer(transformers=[
    ("num", num_pipe, df.select_dtypes(include="number").columns.drop(CLASSIFICATION_TARGET)),
    ("cat", cat_pipe, df.select_dtypes(exclude="number").columns)
])
                                  
model_pipe = Pipeline([
    ("preprocessing", preprocessors),
    ("model", KNeighborsClassifier(n_neighbors=p_n_neighbors, weights=p_weights))
])

# Обучение модели
model_pipe.fit(X_train, y_train.astype('int'))

# Сохранение обученной модели
with open(f_output, "wb") as fd:
    pickle.dump(model_pipe, fd)
