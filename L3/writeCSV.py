import csv


def writeCSV(filename, data):
    fields = ['класс', 'x0', 'x1', 'x2', 'w0', 'w1', 'w2', 'z', 'y', 'верно?']

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

    for row in data:
        writer.writerow(row)

    print(f'Файл {filename} успешно создан.')
