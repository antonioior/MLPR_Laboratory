class Competitor:
    def __init__(self, name, surname, nationality,scores):
        self.name = name
        self.surname = surname
        self.nationality = nationality
        self.scores = scores
        self.final_score = 0

    def calculate_final_score(self):
        self.final_score = sum([score for score in self.scores])
        min_score = min([score for score in self.scores])
        max_score = max([score for score in self.scores])
        self.final_score = self.final_score - min_score - max_score

    def __str__(self):
        return f"{self.name} {self.surname} - Score: {self.final_score:.2f}"