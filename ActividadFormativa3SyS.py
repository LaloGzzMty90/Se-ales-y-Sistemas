"""
Título del Proyecto: Análisis de Filtros Digitales
Autor: Eduardo Israel Gonzalez Flores
Fecha: 16 de julio 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

# Configuración inicial
FS = 1000  # Frecuencia de muestreo
DURATION = 2  # Duración en segundos
t = np.linspace(0, DURATION, int(FS*DURATION), endpoint=False)

# Generar señal de prueba
def generar_senal():
    return 0.8 * np.sin(2*np.pi*10*t) + 1.2 * np.sin(2*np.pi*75*t) + 0.3 * np.random.normal(size=len(t))

senal_original = generar_senal()
# Diseño de filtros
filtros = {
    'Pasa Bajos (30Hz)': signal.butter(4, 30/(0.5*FS), 'low'),
    'Pasa Altos (50Hz)': signal.butter(4, 50/(0.5*FS), 'high'),
    'Pasa Bandas (40-60Hz)': signal.butter(4, np.array([40, 60])/(0.5*FS), 'band')
}

# Aplicar filtros
resultados = {}
for nombre, (b, a) in filtros.items():
    resultados[nombre] = signal.filtfilt(b, a, senal_original)

# Sistema de navegación interactiva
class NavegadorGraficos:
    def __init__(self):
        self.diapositivas = [
            self.diapositiva_original,
            self.diapositiva_pasa_bajos,
            self.diapositiva_pasa_altos,
            self.diapositiva_pasa_bandas,
            self.diapositiva_comparativa
        ]
        self.indice = 0
        self.fig = plt.figure(figsize=(12, 6))
        self.mostrar_diapositiva()

    def mostrar_diapositiva(self):
        self.fig.clf()
        self.diapositivas[self.indice]()
        plt.tight_layout()
        plt.draw()
        print(f"\nDiapositiva {self.indice+1}/{len(self.diapositivas)}")
        print("Presiona: [→] Siguiente, [←] Anterior, [Q] Salir")

    def diapositiva_original(self):
        # Gráfico temporal
        ax1 = self.fig.add_subplot(1, 2, 1)
        ax1.plot(t, senal_original)
        ax1.set_title('Señal Original - Temporal')

        # Gráfico frecuencial
        ax2 = self.fig.add_subplot(1, 2, 2)
        n = len(t)
        f = fftfreq(n, 1/FS)[:n//2]
        ax2.plot(f, np.abs(fft(senal_original))[:n//2])
        ax2.set_title('Señal Original - Frecuencial')

    def diapositiva_pasa_bajos(self):
        self.graficar_filtro('Pasa Bajos (30Hz)')

    def diapositiva_pasa_altos(self):
        self.graficar_filtro('Pasa Altos (50Hz)')

    def diapositiva_pasa_bandas(self):
        self.graficar_filtro('Pasa Bandas (40-60Hz)')

    def diapositiva_comparativa(self):
        # Gráfico comparativo en frecuencia
        n = len(t)
        f = fftfreq(n, 1/FS)[:n//2]

        plt.plot(f, np.abs(fft(senal_original))[:n//2], label='Original')
        for nombre, señal in resultados.items():
            plt.plot(f, np.abs(fft(señal))[:n//2], label=nombre)
        plt.title('Comparación de Filtros - Dominio Frecuencial')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Amplitud')
        plt.legend()
        plt.grid(True)

    def graficar_filtro(self, nombre_filtro):
        señal_filtrada = resultados[nombre_filtro]

        # Gráfico temporal
        ax1 = self.fig.add_subplot(1, 2, 1)
        ax1.plot(t, senal_original, label='Original')
        ax1.plot(t, señal_filtrada, label='Filtrada')
        ax1.set_title(f'{nombre_filtro} - Temporal')
        ax1.legend()

        # Gráfico frecuencial
        ax2 = self.fig.add_subplot(1, 2, 2)
        n = len(t)
        f = fftfreq(n, 1/FS)[:n//2]
        ax2.plot(f, np.abs(fft(senal_original))[:n//2], label='Original')
        ax2.plot(f, np.abs(fft(señal_filtrada))[:n//2], label='Filtrada')
        ax2.set_title(f'{nombre_filtro} - Frecuencial')
        ax2.legend()

    def navegar(self, event):
        if event.key == 'right':
            self.indice = (self.indice + 1) % len(self.diapositivas)
            self.mostrar_diapositiva()
        elif event.key == 'left':
            self.indice = (self.indice - 1) % len(self.diapositivas)
            self.mostrar_diapositiva()
        elif event.key == 'q':
            plt.close()

# Ejecutar el navegador
nav = NavegadorGraficos()
plt.connect('key_press_event', nav.navegar)
plt.show()