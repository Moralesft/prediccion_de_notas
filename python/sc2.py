import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Semilla para reproducibilidad
np.random.seed(42)

# Generar datos ficticios
horas_estudiadas = np.random.randint(1, 10, size=50)  # Generamos 50 valores aleatorios entre 1 y 10 para las horas estudiadas
calificacion_examen = 1 + (6 / 90) * (50 + 10 * horas_estudiadas + np.random.normal(0, 5, size=50))  # Escalamos y ajustamos las calificaciones

# Asegurarnos de que las calificaciones estén dentro del rango de 1 a 7
calificacion_examen = np.clip(calificacion_examen, 1, 7)

# Mostrar los primeros 5 valores de cada conjunto de datos
print("Horas estudiadas:", horas_estudiadas[:5])
print("Calificacion examen:", calificacion_examen[:5])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(horas_estudiadas.reshape(-1, 1), calificacion_examen, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)
print("Error cuadrático medio:", mse)


# Visualizar la relación entre las horas estudiadas y las calificaciones obtenidas
plt.scatter(X_test, y_test, color='blue', label='Datos de prueba')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Línea de regresión')
plt.xlabel('Horas estudiadas')
plt.ylabel('Calificación examen')
plt.title('Relación entre horas estudiadas y calificación del examen')
plt.legend()
plt.show()
