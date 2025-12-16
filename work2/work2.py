# -*- coding: utf-8 -*-

import numpy as np
from numpy.fft import fft, fftshift
import tools
import matplotlib.pyplot as plt


class GaussianDiff:
    ''' Класс с уравнением плоской волны для модулированного гауссова сигнала в дискретном виде

    d - определяет задержку сигнала.
    w - определяет ширину сигнала.
    Nl - количество ячеек на длину волны.
    Sc - число Куранта.
    eps - относительная диэлектрическая проницаемость среды, в которой расположен источник.
    mu - относительная магнитная проницаемость среды, в которой расположен источник.
    '''

    def __init__(self, dt, A, F, Nl, Sc=1.0, eps=1.0, mu=1.0):
        self.A = A
        self.F = F
        self.Nl = Nl
        self.Sc = Sc
        self.eps = eps
        self.mu = mu
        self.dt = dt

    def getE(self, m, q):
        '''
        Расчет поля E в дискретной точке пространства m
        в дискретный момент времени q
        '''
        w = 2 * np.sqrt(np.log(self.A)) / (np.pi * self.F)
        d = w * np.sqrt(np.log(self.A))
        dt = self.dt
        return (np.sin(2 * np.pi / self.Nl * (q * self.Sc - m * np.sqrt(self.eps * self.mu))) *
                np.exp(-(((q - m * np.sqrt(self.eps * self.mu) / self.Sc) - d / dt) / (w / dt)) ** 2))


if __name__ == '__main__':
    d = 0.1e-3

    # Характеристическое сопротивление свободного пространства
    Z0 = 120.0 * np.pi

    # Число Куранта
    Sc = 1.0

    # Магнитная постоянная
    mu0 = np.pi * 4e-7

    # Электрическая постоянная
    eps0 = 8.854187817e-12

    # Скорость света в вакууме
    c = 1.0 / np.sqrt(mu0 * eps0)

    # Расчет "дискретных" параметров моделирования
    dt = d / c * Sc
    print(dt)

    # Время расчета в отсчетах
    maxTime_sec = 20e-9
    maxTime = int(np.ceil(maxTime_sec / dt))

    # Размер области моделирования в отсчетах
    sizeX_m = 2              # УВЕЛИЧЕНО, чтобы влезли все слои варианта 8
    maxSize = int(np.ceil(sizeX_m / d))

    # --- Геометрия слоёв: ВАРИАНТ 8 ---
    # Таблица 1: ε1 = 2.5, d1 = 0.6 м; ε2 = 1.5, d2 = 0.05 м; ε3 = 8, d3 = ∞
    layer_x = 1
    layer_x_DX = int(np.ceil(layer_x / d))

    layer_x_2 = layer_x + 0.6          # конец первого слоя (ε1)
    layer_x_2_DX = int(np.ceil(layer_x_2 / d))

    layer_x_3 = layer_x + 0.6 + 0.05   # конец второго слоя (ε2)
    layer_x_3_DX = int(np.ceil(layer_x_3 / d))

    print(layer_x_DX)
    print(layer_x_2_DX)
    print(layer_x_3_DX)

    # Положение источника в отсчетах
    sourcePosm = 0.5
    sourcePos = int(np.ceil(sourcePosm / d))

    # Датчики для регистрации поля (как в 11‑м)
    probesPos = [int(np.ceil(0.55 / d)), int(np.ceil(0.1 / d))]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Параметры среды (ИЗМЕНЕНЫ под вариант 8)
    eps = np.ones(maxSize)
    eps[layer_x_DX:] = 2.5
    eps[layer_x_2_DX:] = 1.5
    eps[layer_x_3_DX:] = 8

    mu = np.ones(maxSize - 1)

    Ez = np.zeros(maxSize)
    Ezspectrumpad = np.zeros(maxTime)
    Ezspectrumotr = np.zeros(maxTime)

    Hy = np.zeros(maxSize - 1)

    source = GaussianDiff(dt, 100, 5e9, 200)   # как в 11‑м, F = 5 ГГц

    #Коэффициенты для расчета ABC второй степени

    # Sc' для правой границы
    Sc1Right = Sc / np.sqrt(mu[-1] * eps[-1])
    k1Right = -1 / (1 / Sc1Right + 2 + Sc1Right)
    k2Right = 1 / Sc1Right - 2 + Sc1Right
    k3Right = 2 * (Sc1Right - 1 / Sc1Right)
    k4Right = 4 * (1 / Sc1Right + Sc1Right)

    # Ez[0: 2] в предыдущий момент времени (q)
    oldEzLeft1 = np.zeros(3)
    # Ez[0: 2] в пред-предыдущий момент времени (q - 1)
    oldEzLeft2 = np.zeros(3)

    # Ez[-3: -1] в предыдущий момент времени (q)
    oldEzRight1 = np.zeros(3)
    # Ez[-3: -1] в пред-предыдущий момент времени (q - 1)
    oldEzRight2 = np.zeros(3)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel, d, dt)
    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])
    display.drawBoundary(layer_x_DX)
    display.drawBoundary(layer_x_2_DX)
    display.drawBoundary(layer_x_3_DX)

    for q in range(1, maxTime):

        # Расчет компоненты поля H
        Hy[:] = Hy + (Ez[1:] - Ez[:-1]) * Sc / (Z0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (Z0 * mu[sourcePos - 1]) * source.getE(0, q)

        # Расчет компоненты поля E
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy[:-1]) * Sc * Z0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (np.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getE(-0.5, q + 0.5))

        Sc1Left = Sc / np.sqrt(mu[0] * eps[0])
        k1Left = -1 / (1 / Sc1Left + 2 + Sc1Left)
        k2Left = 1 / Sc1Left - 2 + Sc1Left
        k3Left = 2 * (Sc1Left - 1 / Sc1Left)
        k4Left = 4 * (1 / Sc1Left + Sc1Left)

        # Граничные условия ABC второй степени (слева)
        Ez[0] = (k1Left * (k2Left * (Ez[2] + oldEzLeft2[0]) +
                           k3Left * (oldEzLeft1[0] + oldEzLeft1[2] -
                                     Ez[1] - oldEzLeft2[1]) -
                           k4Left * oldEzLeft1[1]) -
                 oldEzLeft2[2])
        oldEzLeft2[:] = oldEzLeft1[:]
        oldEzLeft1[:] = Ez[0: 3]

        # Граничные условия ABC второй степени (справа)
        Ez[-1] = (k1Right * (k2Right * (Ez[-3] + oldEzRight2[-1]) +
                             k3Right * (oldEzRight1[-1] + oldEzRight1[-3] -
                                        Ez[-2] - oldEzRight2[-2]) -
                             k4Right * oldEzRight1[-2]) -
                  oldEzRight2[-3])
        oldEzRight2[:] = oldEzRight1[:]
        oldEzRight1[:] = Ez[-3:]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 1000 == 0:
            display.updateData(display_field, q)

    display.stop()

    # Отображение сигналов, сохраненных в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1, dt)

    # Расчет спектра (КАК В 11‑М ВАРИАНТЕ)
    size = maxTime
    df = 1 / (maxTime * dt)

    start_index_pad = 850 * 10

    for q in range(1, maxTime):
        Ezspectrumpad[q] = probes[0].E[q]
        Ezspectrumotr[q] = probes[1].E[q]
        print(q)

    Ezspectrumpad[start_index_pad:] = [1e-28]

    spectrumpad = fft(Ezspectrumpad)
    spectrumotr = fft(Ezspectrumotr)

    koefotr = spectrumotr / (spectrumpad)
    koefotr = abs(koefotr)
    koefotr = fftshift(koefotr)

    spectrumpad = np.abs(fft(Ezspectrumpad))
    spectrumotr = np.abs(fft(Ezspectrumotr))

    spectrumotr = fftshift(spectrumotr)
    spectrumpad = fftshift(spectrumpad)

    # Расчет частоты
    freq = np.arange(-size / 2 * df, size / 2 * df, df)

    # Отображение спектра
    plt.subplot(1, 1, 1)

    norm_pad = spectrumpad / np.max(spectrumpad)
    norm_otr = spectrumotr / np.max(spectrumpad)  # тот же знаменатель

    plt.plot(freq, norm_pad)
    plt.plot(freq, norm_otr)

    plt.grid()
    plt.xlabel('Частота, Гц')
    plt.ylabel('|S| / |Smax|')
    plt.xlim(0e9, 20e9)

    plt.figure()
    plt.subplot(1, 1, 1)

    plt.plot(freq, koefotr)

    plt.grid()
    plt.ylabel('Г')
    plt.ylim(0, 1)
    plt.xlim(0, 5e9)
    plt.subplots_adjust(wspace=0.4)
    plt.show()