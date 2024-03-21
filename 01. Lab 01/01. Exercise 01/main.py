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

def compute_best_country(competitors):
    countries_dictionary ={}
    distinct_countries = list(set([competitor.nationality for competitor in competitors]))
    for country in distinct_countries:
        countries_dictionary[country] = 0
    for competitor in competitors:
        countries_dictionary[competitor.nationality] += competitor.final_score
    best_country = max(countries_dictionary, key=countries_dictionary.get)
    return best_country, countries_dictionary[best_country]

if __name__ == "__main__":
    print("Name file is", sys.argv[1])
    competitors = load(sys.argv[1])
    [competitor.calculate_final_score() for competitor in competitors]
    best_competitors = sorted(competitors, key=lambda x: x.final_score, reverse=True)[:3]
    print("Final ranking:")
    [print(competitor) for competitor in best_competitors]
    best_country, point = compute_best_country(competitors)
    print(f"\nBest country:\n{best_country} - Total score: {point:.2f}")

