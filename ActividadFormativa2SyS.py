import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Configuración general
plt.style.use('default')
np.set_printoptions(precision=4, suppress=True)

# =============================================
# 1. DEFINICIÓN DE SEÑALES EN EL DOMINIO DEL TIEMPO
# =============================================

def generar_pulso_rectangular(t, ancho=1, amplitud=1, centro=0):
    """
    Genera un pulso rectangular (función de caja)

    Parámetros:
    t: array - Vector de tiempo
    ancho: float - Ancho del pulso
    amplitud: float - Altura del pulso
    centro: float - Centro del pulso en el eje temporal

    Retorna:
    señal: array - Pulso rectangular
    """
    señal = np.zeros_like(t)
    señal[(t >= centro - ancho/2) & (t <= centro + ancho/2)] = amplitud
    return señal

def generar_escalon(t, amplitud=1, inicio=0):
    """
    Genera una función escalón unitario

    Parámetros:
    t: array - Vector de tiempo
    amplitud: float - Altura del escalón
    inicio: float - Punto donde ocurre la transición

    Retorna:
    señal: array - Función escalón
    """
    señal = np.zeros_like(t)
    señal[t >= inicio] = amplitud
    return señal

def generar_senoidal(t, frecuencia=1, amplitud=1, fase=0):
    """
    Genera una señal senoidal

    Parámetros:
    t: array - Vector de tiempo
    frecuencia: float - Frecuencia en Hz
    amplitud: float - Amplitud de la señal
    fase: float - Fase inicial en radianes

    Retorna:
    señal: array - Señal senoidal
    """
    return amplitud * np.sin(2 * np.pi * frecuencia * t + fase)

# =============================================
# 2. CÁLCULO DE LA TRANSFORMADA DE FOURIER
# =============================================

def calcular_transformada_fourier(señal_t, dt):
    """
    Calcula la Transformada Discreta de Fourier (DFT) usando FFT

    Parámetros:
    señal_t: array - Señal en el dominio del tiempo
    dt: float - Paso de tiempo

    Retorna:
    frecuencias: array - Vector de frecuencias
    espectro: array - Transformada de Fourier (compleja)
    """
    N = len(señal_t)
    espectro = np.fft.fft(señal_t) * dt  # Normalizado por dt
    frecuencias = np.fft.fftfreq(N, dt)

    # Reorganizar para tener frecuencias ordenadas
    idx = np.argsort(frecuencias)
    frecuencias = frecuencias[idx]
    espectro = espectro[idx]

    return frecuencias, espectro

# =============================================
# 3. VISUALIZACIÓN DE RESULTADOS
# =============================================

def graficar_señal_y_espectro(t, señal_t, frecuencias, espectro, titulo):
    """
    Grafica la señal en tiempo y su espectro de frecuencia

    Parámetros:
    t: array - Vector de tiempo
    señal_t: array - Señal en dominio temporal
    frecuencias: array - Vector de frecuencias
    espectro: array - Transformada de Fourier
    titulo: str - Título para los gráficos
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, figure=fig)

    # Gráfico de la señal en tiempo
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, señal_t, 'b-', linewidth=2)
    ax1.set_title(f'Señal en el Dominio del Tiempo: {titulo}')
    ax1.set_xlabel('Tiempo [s]')
    ax1.set_ylabel('Amplitud')
    ax1.grid(True)

    # Gráfico de magnitud del espectro
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(frecuencias, np.abs(espectro), 'r-', linewidth=2)
    ax2.set_title('Espectro de Frecuencia (Magnitud)')
    ax2.set_xlabel('Frecuencia [Hz]')
    ax2.set_ylabel('Magnitud')
    ax2.grid(True)
    ax2.set_xlim([-10, 10])  # Limitamos el rango de frecuencias para mejor visualización

    # Gráfico de fase del espectro
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(frecuencias, np.angle(espectro), 'g-', linewidth=2)
    ax3.set_title('Espectro de Frecuencia (Fase)')
    ax3.set_xlabel('Frecuencia [Hz]')
    ax3.set_ylabel('Fase [rad]')
    ax3.grid(True)
    ax3.set_xlim([-10, 10])

    plt.tight_layout()
    plt.show()

# =============================================
# 4. ANÁLISIS DE PROPIEDADES
# =============================================

def verificar_linealidad(t, señal1, señal2, dt):
    """
    Verifica la propiedad de linealidad de la Transformada de Fourier

    Parámetros:
    t: array - Vector de tiempo
    señal1: array - Primera señal
    señal2: array - Segunda señal
    dt: float - Paso de tiempo
    """
    # Combinación lineal en tiempo
    a, b = 2, 3  # Coeficientes arbitrarios
    señal_combinada = a*señal1 + b*señal2

    # Transformadas individuales
    _, espectro1 = calcular_transformada_fourier(señal1, dt)
    _, espectro2 = calcular_transformada_fourier(señal2, dt)

    # Transformada de la combinación lineal
    frecuencias, espectro_combinada = calcular_transformada_fourier(señal_combinada, dt)

    # Combinación lineal en frecuencia
    espectro_lineal = a*espectro1 + b*espectro2

    # Comparación
    error = np.max(np.abs(espectro_combinada - espectro_lineal))
    print(f"Error en verificación de linealidad: {error:.2e}")

    # Gráfico de comparación
    plt.figure(figsize=(12, 6))
    plt.plot(frecuencias, np.abs(espectro_combinada), 'r-', label='TF(a*s1 + b*s2)')
    plt.plot(frecuencias, np.abs(espectro_lineal), 'b--', label='a*TF(s1) + b*TF(s2)')
    plt.title('Verificación de Linealidad de la Transformada de Fourier')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.grid(True)
    plt.xlim([-10, 10])
    plt.show()

def verificar_desplazamiento_tiempo(t, señal, dt, desplazamiento):
    """
    Verifica la propiedad de desplazamiento en tiempo

    Parámetros:
    t: array - Vector de tiempo
    señal: array - Señal original
    dt: float - Paso de tiempo
    desplazamiento: float - Cantidad de desplazamiento en segundos
    """
    # Señal desplazada
    señal_desplazada = np.roll(señal, int(desplazamiento/dt))

    # Transformadas
    frecuencias, espectro = calcular_transformada_fourier(señal, dt)
    _, espectro_desplazado = calcular_transformada_fourier(señal_desplazada, dt)

    # Teoría: TF(s(t-t0)) = e^(-j2πft0) * TF(s(t))
    factor_teorico = np.exp(-2j * np.pi * frecuencias * desplazamiento)
    espectro_teorico = factor_teorico * espectro

    # Comparación
    error = np.max(np.abs(espectro_desplazado - espectro_teorico))
    print(f"Error en verificación de desplazamiento temporal: {error:.2e}")

    # Gráfico de comparación (fase)
    plt.figure(figsize=(12, 6))
    plt.plot(frecuencias, np.angle(espectro_desplazado), 'r-', label='TF(señal desplazada)')
    plt.plot(frecuencias, np.angle(espectro_teorico), 'b--', label='Teoría')
    plt.title('Verificación de Desplazamiento Temporal: Fase del Espectro')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Fase [rad]')
    plt.legend()
    plt.grid(True)
    plt.xlim([-10, 10])
    plt.show()

def verificar_escalamiento_frecuencia(t, señal, dt, factor):
    """
    Verifica la propiedad de escalamiento en frecuencia

    Parámetros:
    t: array - Vector de tiempo
    señal: array - Señal original
    dt: float - Paso de tiempo
    factor: float - Factor de escalamiento
    """
    # Señal escalada en tiempo (comprimida si factor > 1)
    t_escalado = t * factor
    señal_escalada = np.interp(t, t_escalado, señal, left=0, right=0)

    # Transformadas
    frecuencias, espectro = calcular_transformada_fourier(señal, dt)
    _, espectro_escalado = calcular_transformada_fourier(señal_escalada, dt)

    # Teoría: TF(s(at)) = (1/|a|) * TF(f/a)
    frecuencias_teoricas = frecuencias * factor
    espectro_teorico = (1/np.abs(factor)) * np.interp(frecuencias, frecuencias_teoricas, np.abs(espectro), left=0, right=0)

    # Comparación (solo magnitud)
    error = np.max(np.abs(np.abs(espectro_escalado) - espectro_teorico))
    print(f"Error en verificación de escalamiento: {error:.2e}")

    # Gráfico de comparación
    plt.figure(figsize=(12, 6))
    plt.plot(frecuencias, np.abs(espectro_escalado), 'r-', label='TF(señal escalada)')
    plt.plot(frecuencias, espectro_teorico, 'b--', label='Teoría')
    plt.title(f'Verificación de Escalamiento (factor={factor}): Magnitud del Espectro')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.grid(True)
    plt.xlim([-10, 10])
    plt.show()

# =============================================
# 5. SIMULACIÓN PRINCIPAL
# =============================================

def main():
    # Configuración de parámetros
    dt = 0.01  # Paso de tiempo [s]
    t = np.arange(-5, 5, dt)  # Vector de tiempo de -5 a 5 segundos

    # Generación de señales
    pulso = generar_pulso_rectangular(t, ancho=2, amplitud=1, centro=0)
    escalon = generar_escalon(t, amplitud=1, inicio=0)
    seno = generar_senoidal(t, frecuencia=2, amplitud=1, fase=0)

    # Análisis de cada señal
    señales = {
        'Pulso Rectangular': pulso,
        'Función Escalón': escalon,
        'Señal Senoidal': seno
    }

    for nombre, señal in señales.items():
        print(f"\nAnalizando señal: {nombre}")
        frecuencias, espectro = calcular_transformada_fourier(señal, dt)
        graficar_señal_y_espectro(t, señal, frecuencias, espectro, nombre)

    # Verificación de propiedades
    print("\nVerificación de propiedades:")

    # Linealidad (combinación de pulso y seno)
    print("\n1. Propiedad de Linealidad:")
    verificar_linealidad(t, pulso, seno, dt)

    # Desplazamiento en tiempo (pulso rectangular)
    print("\n2. Propiedad de Desplazamiento en Tiempo:")
    verificar_desplazamiento_tiempo(t, pulso, dt, desplazamiento=1)

    # Escalamiento en frecuencia (función escalón)
    print("\n3. Propiedad de Escalamiento en Frecuencia:")
    verificar_escalamiento_frecuencia(t, escalon, dt, factor=2)

if __name__ == "__main__":
    main()