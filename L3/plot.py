from matplotlib import pyplot as plt


def plotData(Xs, Os):
    x1 = [Xs[i][0] for i in range(len(Xs))]
    y1 = [Xs[i][1] for i in range(len(Xs))]

    x2 = [Os[i][0] for i in range(len(Os))]
    y2 = [Os[i][1] for i in range(len(Os))]

    plt.plot(x1, y1, 'rx', label='+')
    plt.plot(x2, y2, 'bo', label='-')

    plt.axhline(y=0, color='k', linestyle='-', linewidth=1)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=1)

    plt.xlim(-2, 3)
    plt.ylim(-2, 3)

    # Отображение сетки
    plt.grid(True)

    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.title('График с двумя видами точек')
    plt.legend()
