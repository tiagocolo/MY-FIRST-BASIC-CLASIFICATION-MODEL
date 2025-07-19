import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

def evaluacion_caracteristicas(mensaje):
  signos_de_exclamacion = mensaje.count("!")
  mensaje_limpio = mensaje.replace(",","").replace("!","").replace(".","")
  cantidad_palabras = len(mensaje_limpio.split())
  return np.array([cantidad_palabras, signos_de_exclamacion])



X_spam_val = [
    [6, 10], [9, 12], [5, 8], [6, 11], [10, 13], [6, 9], [7, 11], [8, 13], [5, 8], [6, 10]
]

y_spam_val = [1] * len(X_spam_val)

X_no_spam_val = [
    [32, 1], [45, 0], [38, 2], [51, 1], [27, 0], [32, 2], [39, 1], [41, 0], [45, 1], [37, 2]
]

y_no_spam_val = [0] * len(X_no_spam_val)

x_val = np.array(X_spam_val + X_no_spam_val, dtype=np.float32)
y_val = np.array(y_spam_val + y_no_spam_val, dtype=np.float32)

normalizer_val = tf.keras.layers.Normalization(axis=-1)
normalizer_val.adapt(x_val)

x_normalized_val = normalizer_val(x_val)


def generar_datos(n_spam=5000, n_no_spam=5000, seed=42):
    np.random.seed(seed)
    datos = []
    etiquetas = []

    # Generar NO SPAM
    for _ in range(n_no_spam):
        a = np.random.randint(4, 5000)  # Palabras
        b = np.random.randint(0, a // 4 + 1)  # Exclamaciones pequeñas
        datos.append([a, b])
        etiquetas.append(0)

    # Generar SPAM
    for _ in range(n_spam):
        a = np.random.randint(4, 5000)
        b = np.random.randint(a // 4 + 1, a + 5)  # Exclamaciones grandes
        datos.append([a, b])
        etiquetas.append(1)

    return np.array(datos, dtype=np.float32), np.array(etiquetas)

# Usar la función
X, y = generar_datos()

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X)

# Verificar
print("Primeras 5 entradas:\n", X[:5])
print("Primeras 5 etiquetas:\n", y[:5])

normalizer
capa_oculta= tf.keras.layers.Dense(units=3, activation='relu')
capay = tf.keras.layers.Dense(units=1, activation = 'sigmoid')


model = tf.keras.Sequential([
    normalizer,
    capa_oculta,
    capay
     ])


model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])


history = model.fit(X, y, epochs=1000, batch_size=16, verbose=False, validation_data=(x_normalized_val, y_val))

print("Modelo entrenado!")

plt.plot(history.history['loss'])
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.show()

mensaje = input("Ingrese un mensaje: ")

resultado_prediccion=model.predict(np.array([evaluacion_caracteristicas(mensaje)]))
if resultado_prediccion> 0.7:
  print('El correo es spam')
else:
  print('El correo no es spam')
