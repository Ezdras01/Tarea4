import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Tus datos
x = np.array(range(1,30)) 
y = np.array([20.214, 18.413, 15.754, 14.125, 14.024, 13.226, 15.458, 14.547, 14.754, 13.536, 12.425, 10.543, 10.058, 9.135, 7.698, 5.564, 4.213, 3.896, 6.012, 7.894, 10.214, 12.266, 15.124, 16.989, 19.014, 21.254, 22.887, 24.364, 25.898])

# Ajuste de los mínimos cuadrados
model = LinearRegression()
model.fit(x.reshape(-1,1), y)

# Predicción
y_pred = model.predict(x.reshape(-1,1))

# Visualización
plt.scatter(x, y, color='blue')
for i in range(len(x) - 1):
    plt.plot(x[i:i+2], y[i:i+2], color='red')
plt.title('Mínimos cuadrados segmentados')
plt.xlabel('x')
plt.ylabel('y')
plt.show()