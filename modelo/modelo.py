# import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import pickle


iris = datasets.load_iris()

X = iris.data
y = iris.target

# one = OneHotEncoder(sparse=False)
# y = one.fit_transform(y.reshape(-1, 1))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


classifier = LogisticRegression()
classifier.fit(X_train, y_train)
result = classifier.score(X_test, y_test)

pickle.dump(classifier, open('LRClassifier.pkl', 'wb'))


# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Input(shape=(X.shape[1])))
# model.add(tf.keras.layers.Dense(10, activation="relu"))
# model.add(tf.keras.layers.Dense(10, activation="relu"))
# model.add(tf.keras.layers.Dense(3, activation='softmax'))
# model.compile(loss='categorical_crossentropy',
#               metrics=['accuracy'], optimizer='rmsprop')

# history = model.fit(X_train, y_train, epochs=100,
#                     validation_data=(X_test, y_test))

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))


# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Metrica de erro')
# plt.ylabel('Erro')
# plt.xlabel('Epoca')
# plt.legend(['Treinamento', 'Validacao'])
# plt.show()


# model.save('model.h5')

# pred = np.round(model.predict(X_test))
# pred_train = np.round(model.predict(X_train))
