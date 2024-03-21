import sys 
from competitor import *

def load(nameFile):
    rows = open(nameFile, 'r')
    competitors = []
    for row in rows:
        row = row.strip().split(' ')
        name = row[0]
        surname = row[1]
        nationality = row[2]
        scores = [float(score) for score in row[3:]]
        competitor = Competitor(name, surname, nationality, scores)
        competitors.append(competitor)
    return competitors

if __name__ == "__main__":
    print("Name file is", sys.argv[1])
    competitors = load(sys.argv[1])
    [competitor.calculate_final_score() for competitor in competitors]
    best_competitors = sorted(competitors, key=lambda x: x.final_score, reverse=True)[:3]
    print("Final ranking:\n")
    [print(competitor) for competitor in best_competitors]
