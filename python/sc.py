import numpy as np

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
