def load_data():
    lInf = []
    f = open('../Data/inferno.txt', encoding="ISO-8859-1")
    for line in f:
        lInf.append(line.strip())
    f.close()
    lPur = []
    f = open('../Data/purgatorio.txt', encoding="ISO-8859-1")
    for line in f:
        lPur.append(line.strip())
    f.close()
    lPar = []
    f = open('../Data/paradiso.txt', encoding="ISO-8859-1")
    for line in f:
        lPar.append(line.strip())
    f.close()
    return lInf, lPur, lPar
