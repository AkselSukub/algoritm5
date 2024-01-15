#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import math
import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import timeit


def bubble_sort(arr):
    n = len(arr)
    for i in range(n-1):
        for j in range(n-1-i):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr


def coeffs(xs, ys):
    n = len(xs)
    s1 = sum(xs)
    s2 = sum(x ** 2 for x in xs)
    s3 = sum(x ** 3 for x in xs)
    s4 = sum(x ** 4 for x in xs)
    sy0 = sum(ys)
    sy1 = sum(xs[i] * ys[i] for i in range(n))
    sy2 = sum((xs[i] ** 2) * ys[i] for i in range(n))
    matrixx = [[n, s1, s2], [s1, s2, s3], [s2, s3, s4]]
    matrixy = [[sy0], [sy1], [sy2]]
    x = np.linalg.solve(matrixx, matrixy)
    return x[2][0], x[1][0], x[0][0]


def create_graph(x, y, namegraph):
    plt.scatter(x, y, s=5)
    plt.title(namegraph)
    plt.xlabel("Размер массива")
    plt.ylabel("Время работы функции")


if __name__ == '__main__':
    count = 50
    x1 = [i for i in range(10, count*10+1, 10)]
    x2 = [[i]*30 for i in x1]
    xgraph = list(itertools.chain.from_iterable(x2))
    randmax = 1000000
    timesred = []
    timehud = []

    for i in xgraph:
        sred = [rnd.randint(0, randmax) for j in range(i)]
        timesred.append(timeit.timeit(lambda: bubble_sort(sred), number=1))

    for i in x1:
        hud = [j for j in range(i, 0, -1)]
        timehud.append(timeit.timeit(lambda: bubble_sort(hud), number=1))

    timedel = [timesred[i: i+30] for i in range(0, len(timesred), 30)]
    e = [(1/30*(sum(timedel[i]))) for i in range(len(timedel))]

    sigmavalue = [sum([(timedel[k][j] - e[k])**2 for j in range(30)])
                for k in range(count)]
    sigma = [math.sqrt(1/29*sigmavalue[k]) for k in range(count)]
    a, b, c = coeffs(x1, e)
    ysred = a * np.array(x1) ** 2 + b * np.array(x1) + c

    a, b, c = coeffs(x1, timehud)
    yhud = a * np.array(x1) ** 2 + b * np.array(x1) + c

    name = "Средний случай исходные данные"
    plt.figure(name)
    create_graph(xgraph, timesred, name)

    name = "Средний случай средние значения с отклонениями и параболой"
    plt.figure(name)
    plt.errorbar(x1, e, yerr=sigma, fmt='none', capsize=2)
    plt.plot(x1, ysred, color="green", linewidth=2)
    create_graph(x1, e, name)

    name = "Худший случай"
    plt.figure(name)
    plt.plot(x1, yhud, color="green", linewidth=2)
    create_graph(x1, timehud, name)

    plt.show()