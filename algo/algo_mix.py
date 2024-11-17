import csv


def random_lcg(x_n, a=1664525, c=5779, m=32767):
    return (a * x_n + c) % m


def random_ud(a, b, x=123):
    return round(a + ((b - a) * (random_lcg(x) / 32767.0)))


def shuffle(data_):
    for i in range(len(data_)):
        random_raw = random_ud(1, len(data_))
        change = data_[random_raw]
        data_[random_raw] = data_[i]
        data_[i] = change
    return data_


def write_csv(path, legends, data_):
    with open(path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(legends)
        for r in data_:
            writer.writerow(r)


def read_csv(path):
    dataset = []
    with open(path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            dataset.append(row)
    return dataset[1:-1], dataset[0]


data, legend = read_csv('Loan_Default_Normalized.csv')
data = shuffle(data)
write_csv('Loan_Default_Normalized_Mixed.csv', legend, data)
