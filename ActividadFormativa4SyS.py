import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Configuración inicial
FS = 44100  # Frecuencia de muestreo (Hz)
DURATION = 0.1  # Duración de la señal (s)
t = np.linspace(0, DURATION, int(FS * DURATION), endpoint=False)

# 1. Señal de mensaje (baja frecuencia)
f_msg = 50  # Frecuencia del mensaje (Hz)
A_msg = 1.0  # Amplitud del mensaje
mensaje = A_msg * np.sin(2 * np.pi * f_msg * t)

# 2. Señal portadora (alta frecuencia)
f_port = 2000  # Frecuencia portadora (Hz)
A_port = 1.0
portadora = A_port * np.sin(2 * np.pi * f_port * t)

# 3. Modulación AM
A_mod = 0.5  # Índice de modulación (0 < A_mod ≤ 1)
senal_AM = (1 + A_mod * mensaje) * portadora

# 4. Visualización - Dominio del tiempo
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, mensaje, 'b')
plt.title("Señal de Mensaje (50 Hz)")
plt.xlabel("Tiempo [s]")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, portadora, 'r')
plt.title("Señal Portadora (2000 Hz)")
plt.xlabel("Tiempo [s]")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, senal_AM, 'g')
plt.title("Señal Modulada en AM")
plt.xlabel("Tiempo [s]")
plt.grid(True)

plt.tight_layout()
plt.show()

# 5. Análisis en frecuencia
n = len(t)
freq = fftfreq(n, 1/FS)[:n//2]

plt.figure(figsize=(12, 4))
plt.plot(freq, np.abs(fft(senal_AM)[:n//2]))
plt.title("Espectro de la Señal AM")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(0, 3000)
plt.grid(True)
plt.show()

# 6. Simulación de ruido
ruido = 0.2 * np.random.normal(size=len(t))
senal_ruidosa = senal_AM + ruido

plt.figure(figsize=(12, 4))
plt.plot(t, senal_ruidosa, 'm')
plt.title("Señal AM con Ruido")
plt.xlabel("Tiempo [s]")
plt.grid(True)
plt.show()