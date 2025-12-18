# -*- coding: utf-8 -*-
import numpy as np
from numpy.fft import fft, fftshift
import tools
import matplotlib.pyplot as plt


# =====================================================
# КЛАСС ИСТОЧНИКА - ГАУССОВ ИМПУЛЬС (БЕЗ МОДУЛЯЦИИ)
# =====================================================


class GaussianPlaneWave:
    """Немодулированный гауссов импульс, как в tr.py"""
    def __init__(self, dt, d, w, Sc=1.0, eps=1.0, mu=1.0):
        self.d = d          # задержка в секундах
        self.dt = dt        # шаг по времени
        self.w = w          # ширина импульса в секундах
        self.Sc = Sc
        self.eps = eps
        self.mu = mu

    def getE(self, m, q):
        # q — индекс времени, m — индекс ячейки
        return np.exp(
            -(((q - m * np.sqrt(self.eps * self.mu) / self.Sc)
               - self.d / self.dt) / (self.w / self.dt)) ** 2
        )


# =====================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================


def create_layer_boundaries(layer_x_start, d1, d2, d):
    """Физические координаты границ слоёв -> индексы сетки
    вакуум | eps1 (d1) | eps2 (d2) | eps3 (полупространство)
    """
    layer1_x = layer_x_start           # начало eps1
    layer2_x = layer1_x + d1           # граница вакуум / eps1
    layer3_x = layer2_x + d2           # граница eps1 / eps2 (дальше eps3)

    layer1_DX = int(np.ceil(layer1_x / d))
    layer2_DX = int(np.ceil(layer2_x / d))
    layer3_DX = int(np.ceil(layer3_x / d))

    return [layer1_DX, layer2_DX, layer3_DX]


def setup_abc_coefficients(Sc, mu, eps):
    """Коэффициенты ABC (Mur 2‑го порядка)"""
    # левая граница
    Sc1Left = Sc / np.sqrt(mu[0] * eps[0])
    k1Left = -1 / (1 / Sc1Left + 2 + Sc1Left)
    k2Left = 1 / Sc1Left - 2 + Sc1Left
    k3Left = 2 * (Sc1Left - 1 / Sc1Left)
    k4Left = 4 * (1 / Sc1Left + Sc1Left)

    # правая граница
    Sc1Right = Sc / np.sqrt(mu[-1] * eps[-1])
    k1Right = -1 / (1 / Sc1Right + 2 + Sc1Right)
    k2Right = 1 / Sc1Right - 2 + Sc1Right
    k3Right = 2 * (Sc1Right - 1 / Sc1Right)
    k4Right = 4 * (1 / Sc1Right + Sc1Right)

    return (k1Left, k2Left, k3Left, k4Left,
            k1Right, k2Right, k3Right, k4Right)


# =====================================================
# ОСНОВНАЯ ПРОГРАММА — ВАРИАНТ 8
# =====================================================


if __name__ == '__main__':
    # -------- дискретизация --------
    d = 0.3e-3                 # шаг по X, 0.3 мм
    Z0 = 120.0 * np.pi
    Sc = 1.0
    mu0 = np.pi * 4e-7
    eps0 = 8.854187817e-12
    c = 1.0 / np.sqrt(mu0 * eps0)
    dt = d / c * Sc
    print(f"dt = {dt}")

    maxTime_sec = 70e-9
    maxTime = int(np.ceil(maxTime_sec / dt))
    sizeX_m = 2.0
    maxSize = int(np.ceil(sizeX_m / d))

    # -------- параметры варианта 8 --------
    # eps1=2.5, d1=0.6 м; eps2=1.5, d2=0.05 м; eps3=8; слева вакуум eps4=1
    eps1, eps2, eps3, eps4 = 2.5, 1.5, 8.0, 1.0
    d1, d2 = 0.6, 0.05

    # сдвиг структуры
    layer_x_start = 0.9

    layer1_DX, layer2_DX, layer3_DX = create_layer_boundaries(
        layer_x_start, d1, d2, d
    )

    # -------- источник и пробы --------
    sourcePosm = 0.1
    sourcePos = int(np.ceil(sourcePosm / d))
    probesPos = [int(np.ceil(0.3 / d)), int(np.ceil(0.01 / d))]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # -------- профиль среды --------
    # вакуум -> eps1 (d1) -> eps2 (d2) -> eps3 до конца
    eps = np.ones(maxSize) * eps4
    eps[layer1_DX:layer2_DX] = eps1
    eps[layer2_DX:layer3_DX] = eps2
    eps[layer3_DX:] = eps3
    mu = np.ones(maxSize - 1)

    # -------- поля --------
    Ez = np.zeros(maxSize)
    Ezspectrumpad = np.zeros(maxTime)
    Ezspectrumotr = np.zeros(maxTime)
    Hy = np.zeros(maxSize - 1)

    # -------- гауссов импульс --------
    A_0 = 100.0
    A_max = 100.0
    F_max = 5e9
    w_g = np.sqrt(np.log(A_max)) / (np.pi * F_max)
    d_g = w_g * np.sqrt(np.log(A_0)) * 2
    print(f"w_g = {w_g}")
    print(f"d_g = {d_g}")

    source = GaussianPlaneWave(dt, d_g, w_g, Sc, eps[sourcePos], mu[sourcePos])

    # -------- ABC --------
    (k1Left, k2Left, k3Left, k4Left,
     k1Right, k2Right, k3Right, k4Right) = setup_abc_coefficients(Sc, mu, eps)

    oldEzLeft1 = np.zeros(3)
    oldEzLeft2 = np.zeros(3)
    oldEzRight1 = np.zeros(3)
    oldEzRight2 = np.zeros(3)

    # -------- визуализация --------
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin, display_ymax = -1.1, 1.1
    display = tools.AnimateFieldDisplay(
        maxSize, display_ymin, display_ymax, display_ylabel, d, dt
    )
    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])
    display.drawBoundary(layer1_DX)   # вакуум / eps1
    display.drawBoundary(layer2_DX)   # eps1 / eps2
    display.drawBoundary(layer3_DX)   # eps2 / eps3

    # -------- основной цикл FDTD --------
    for q in range(1, maxTime):
        # H
        Hy[:] = Hy + (Ez[1:] - Ez[:-1]) * Sc / (Z0 * mu)
        Hy[sourcePos - 1] -= Sc / (Z0 * mu[sourcePos - 1]) * source.getE(0, q)

        # E
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy[:-1]) * Sc * Z0 / eps[1:-1]

        # жёсткий источник
        Ez[sourcePos] += (Sc / np.sqrt(eps[sourcePos] * mu[sourcePos]) *
                          source.getE(-0.5, q + 0.5))

        # ABC слева
        Ez[0] = (k1Left * (k2Left * (Ez[2] + oldEzLeft2[0]) +
                           k3Left * (oldEzLeft1[0] + oldEzLeft1[2]
                                     - Ez[1] - oldEzLeft2[1]) -
                           k4Left * oldEzLeft1[1]) - oldEzLeft2[2])
        oldEzLeft2[:] = oldEzLeft1[:]
        oldEzLeft1[:] = Ez[0:3]

        # ABC справа
        Ez[-1] = (k1Right * (k2Right * (Ez[-3] + oldEzRight2[-1]) +
                             k3Right * (oldEzRight1[-1] + oldEzRight1[-3]
                                        - Ez[-2] - oldEzRight2[-2]) -
                             k4Right * oldEzRight1[-2]) - oldEzRight2[-3])
        oldEzRight2[:] = oldEzRight1[:]
        oldEzRight1[:] = Ez[-3:]

        # пробы
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 1000 == 0:
            display.updateData(display_field, q)

    display.stop()
    tools.showProbeSignals(probes, -1.1, 1.1, dt)

    # -------- спектральный анализ --------
    size = maxTime
    df = 1 / (maxTime * dt)
    start_index_pad = 250 * 10

    for q in range(1, maxTime):
        Ezspectrumpad[q] = probes[0].E[q]
        Ezspectrumotr[q] = probes[1].E[q]

    Ezspectrumpad[start_index_pad:] = 1e-28

    spectrumpad = fft(Ezspectrumpad)
    spectrumotr = fft(Ezspectrumotr)
    koefotr = np.abs(fftshift(spectrumotr / (spectrumpad + 1e-30)))

    spectrumpad = np.abs(fftshift(fft(Ezspectrumpad)))
    spectrumotr = np.abs(fftshift(fft(Ezspectrumotr)))
    freq = np.arange(-size / 2 * df, size / 2 * df, df)
    norm = np.max(spectrumpad)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(freq / 1e9, spectrumpad / norm, label='Падающий')
    plt.plot(freq / 1e9, spectrumotr / norm, label='Отражённый')
    plt.grid()
    plt.xlabel('f, ГГц')
    plt.ylabel('|S| / |Smax падающего|')
    plt.xlim(0, 5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(freq / 1e9, koefotr)
    plt.grid()
    plt.xlabel('f, ГГц')
    plt.ylabel('|Γ|')
    plt.ylim(0, 1)
    plt.xlim(0, 5)
    plt.tight_layout()
    plt.show()
