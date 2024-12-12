
def read(path):
    dataset = []
    with open(path, mode='r', encoding='utf-8') as file:
        for row in file.readlines():
            if row.find(";;") == -1 and row.find(";?;") == -1 and row.find(";") != 0 and not ('�' in row):
                dataset.append(row)

    return dataset

def write(path, _data):
    with open(path, mode='w') as f:
        for r in _data:
            f.write(r)

write("D:\IdeaProjects\\bml2\coursework\loan_data_no_missing.csv", read("D:\IdeaProjects\\bml2\coursework\loan_data.csv"))
