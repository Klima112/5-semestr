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
    print (dt)
    # Время расчета в отсчетах
    maxTime_sec = 40e-9
    maxTime = int(np.ceil(maxTime_sec / dt))
    
    # Размер области моделирования в отсчетах
    sizeX_m = 4
    maxSize = int(np.ceil(sizeX_m / d))
    layer_x = 0
    layer_x_DX=int(np.ceil(layer_x / d))
    layer_x_end=4
    layer_x_end_DX=int(np.ceil(layer_x_end / d))

    # Положение источника в отсчетах
    sourcePos = 0.15
    sourcePos = int(np.ceil(sourcePos / d))

    # Датчики для регистрации поля
    probesPos = 0.60
    probesPos = [int(np.ceil(probesPos / d))]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Параметры среды
    eps = np.ones(maxSize)
    eps[layer_x_DX:] = 9
    eps[layer_x_end_DX:] = 8
    mu = np.ones(maxSize-1)
    
    Ez = np.zeros(maxSize)
    Ezspectrum = np.zeros(maxTime)
    Hy = np.zeros(maxSize - 1)

    source = GaussianDiff(dt, 100, 3.5e9, 1700)

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
                                        display_ylabel,d,dt)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])
    display.drawBoundary(layer_x_DX)
    display.drawBoundary(layer_x_end_DX)
    

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

        # Граничные условия ABC второй степени (справа)
        Ez[-1] = (k1Right * (k2Right * (Ez[-3] + oldEzRight2[-1]) +
                             k3Right * (oldEzRight1[-1] + oldEzRight1[-3] - Ez[-2] - oldEzRight2[-2]) -
                             k4Right * oldEzRight1[-2]) - oldEzRight2[-3])

        oldEzRight2[:] = oldEzRight1[:]
        oldEzRight1[:] = Ez[-3:]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)
            


        if q % 6000 == 0:
            display.updateData(display_field, q)


    display.stop()

    for q in range(1, maxTime):
        Ezspectrum[q]=probes[0].E[q]
    

    # Отображение сигналов, сохраненных в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1,dt)

    # Расчет спектра
    size = maxTime

    df = 1 / (size * dt)
    print(Ezspectrum)
    
    spectrum = np.abs(fft(Ezspectrum))
    spectrum = fftshift(spectrum)

    # Расчет частоты
    freq = np.arange(-size / 2 * df, size / 2 * df, df)
    # Отображение спектра
    plt.subplot(1, 2, 2)
    plt.plot(freq, spectrum / np.max(spectrum))
    plt.grid()
    plt.xlabel('Частота, Гц')
    plt.ylabel('|S| / |Smax|')
    plt.xlim(0e9, 3.5e9)

    plt.subplots_adjust(wspace=0.4)
    plt.show()
