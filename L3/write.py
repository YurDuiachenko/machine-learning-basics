def write_csv(path, legends, _data):
    with open(path, mode='w') as f:
        f.write(legends)
        for r in _data:
            f.write("\n")
            f.write(r)
