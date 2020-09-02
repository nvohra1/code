class Student:

    def __init__(self, name, major, gpa, isOnProbation):
        self.name = name
        self.major = major
        self.gpa = gpa
        self.isOnProbation =isOnProbation
    def info(self):
        print(self.name)
        print(self.major)
        print(self.gpa)
        print(self.isOnProbation)