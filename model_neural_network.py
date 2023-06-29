import pandas as pd
import numpy as np
import tensorflow as tf

raw_data = pd.read_csv('bank_data.csv')
df = raw_data.copy()
df = df.drop(df[['RowNumber', 'CustomerId', 'Surname']], axis = 1)
df = df.dropna()

x = df.iloc[:, : -1]
y = df.iloc[:, -1]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 365)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

hidden_layers_size = 50
output_size = 1

ann = tf.keras.models.Sequential([tf.keras.layers.Dense(hidden_layers_size, activation = 'relu'),
                                  tf.keras.layers.Dense(hidden_layers_size, activation = 'relu'),
                                  tf.keras.layers.Dense(hidden_layers_size, activation = 'relu'),
                                  tf.keras.layers.Dense(output_size, activation = 'sigmoid')
                                ])
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(x_train, y_train, batch_size = 72, epochs = 100)

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.values.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)