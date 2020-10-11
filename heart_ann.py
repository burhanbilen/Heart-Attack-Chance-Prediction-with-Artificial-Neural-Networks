import pandas as pd
from numpy import array
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

df = pd.read_csv('heart.csv')

X = df.iloc[:,0:13].values
y = df.iloc[:,13:14].values
y = array(y).reshape(len(y),1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = Sequential()
model.add(Dense(16, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(8, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 200, batch_size = 32, validation_data = (X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy: ", accuracy)
print("Loss: ", loss)   

pred = model.predict(X_test[:])
target = [1 if i > 0.5 else 0 for i in pred]

print(confusion_matrix(y_test, target))
