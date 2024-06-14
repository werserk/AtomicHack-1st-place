def get_box_cords(path_to_label: str) -> dict:
    defects = {}
    with open(path_to_label) as f:
        for line in f:
            line = line.split()
            if line == []:
                return {}

            if line[0] in defects:
                defects[line[0]].append([float(i) for i in line[1:]])
            else:
                defects[line[0]] = [[float(i) for i in line[1:]]]

    return defects