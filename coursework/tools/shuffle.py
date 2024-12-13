import csv


def random_lcg(x_n, a=1664525, c=5779, m=32767):
    return (a * x_n + c) % m


def random_ud(a, b, x=123):
    return round(a + ((b - a) * (random_lcg(x) / 32767.0)))


def shuffle(_data):
    for i in range(len(_data)):
        random_raw = random_ud(1, len(_data))
        change = _data[random_raw]
        _data[random_raw] = _data[i]
        _data[i] = change
    return _data


def write_csv(path, legends, _data):
    with open(path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(legends)
        for r in _data:
            writer.writerow(r)


def read_csv(path):
    dataset = []
    with open(path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            dataset.append(row)
    return dataset[1:-1], dataset[0]


data, legend = read_csv('../data/Loan_Default_Normalized_Cleaned.csv')
data = shuffle(data)

write_csv('../data/Loan_Default_Normalized_Cleaned_Mixed.csv', legend, data)
