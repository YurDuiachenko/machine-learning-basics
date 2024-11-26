import numpy as np

# (n,) или (1, n) - горизонтальный
# (n, 1) - вертикальный

# 3x1
a = np.array(
    [[1],
     [3],
     [5]]
)

# 3x1
b = np.array(
    [[4],
     [6],
     [8]]
)

print(a.size, a.ndim, a.shape)
print(b.size, b.ndim, b.shape)
print()


def sc_pr_loop(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


def sc_pr_elements(a, b):
    return np.sum(a * b)


def sc_pro_dot(a, b):
    return np.dot(a, b)


print("Скалярное произведение с использованием цикла:",
      sc_pr_loop(a, b))

print("Скалярное произведение с использованием поэлементного умножения:",
      sc_pr_elements(a, b))

print("Скалярное произведение с использованием dot():",
      sc_pro_dot(a.T, b))
