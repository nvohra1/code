import random

metersInKilometer = 100
beatles = ["John Lennon", "Paul McCartney","George Harrison"]

def getFileExt(filename):
    return filename[filename.index(".") + 1:]

def rollDice(num):
    print("I am in")
    ret = random.randint(1,num)
    print(ret)
    return ret

print(rollDice(10))