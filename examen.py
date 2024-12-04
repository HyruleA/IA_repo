import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''Datos del Ejercicio + Generados'''
X = np.array([
    [0.9, 0.8, 0.2],
    [0.7, 0.6, 0.5],
    [0.4, 0.4, 0.8],
    [0.8, 0.9, 0.3],
    [0.5, 0.7, 0.6],
    [0.3, 0.5, 0.9],
    [0.4, 0.8, 0.7],
    [0.5, 0.4, 0.9],
    [0.7, 0.6, 0.3],
    [0.3, 0.8, 0.3],
    [0.2, 0.6, 0.8],
    [0.2, 0.7, 0.2],
    [0.2, 0.7, 0.6],
    [0.3, 0.5, 0.5],
    [0.6, 0.2, 0.2],
    [0.6, 0.3, 0.8],
    [0.7, 0.5, 0.4],
    [0.8, 0.2, 0.8],
    [0.8, 0.2, 0.4],
    [0.6, 0.8, 0.7],
    [0.2, 0.4, 0.8],
    [0.7, 0.7, 0.7],
    [0.7, 0.2, 0.2],
    [0.3, 0.2, 0.2],
    [0.3, 0.8, 0.8],
    [0.6, 0.8, 0.6],
    [0.5, 0.2, 0.7]
])
Y = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0] 
])


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_dim=3, activation='relu'), 
    tf.keras.layers.Dense(3, activation='softmax')  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=100, batch_size=2, verbose=1)

loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Precisión: {accuracy:.2f}")

'''Grafica con mathplotlib'''
def plot_decision_boundary(model, X, Y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())])
    Z = np.argmax(Z, axis=1).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=np.argmax(Y, axis=1), edgecolor='k')
    plt.xlabel('Historial de pagos')
    plt.ylabel('Ingresos mensuales')
    plt.show()

plot_decision_boundary(model, X, Y)


