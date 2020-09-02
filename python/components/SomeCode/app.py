import usefulTools
from student import Student
from usefulTools import getFileExt

print("Hello World")
print("   /|")
print("  / |")
print(" /  |")
print("/___|")

def sayHi(arg1, arg2):
    print("Function Start=====  " + str(arg1) + "...." + str(arg2))
    ret = arg1**arg2
    return ret

print("1")
aaya = sayHi(5, 7)

print(aaya)

isMale = True #True
isTall = True
isDead = False

if(isMale and isTall and not(isDead)):
    print("You are a Male")
else:
    print("You are not a Male")

def maxNum(num1,num2,num3):
    if(num1 >= num2 and num1>=num3):
        return num1
    elif (num2 >= num1 and num2>=num1):
        return num2
    else:
        return num3

print("Max is " + str(maxNum(20,10,15)))


def cal(num1, num2, operator):
    if operator == "+":
        return num1 + num2
    elif operator == "-":
        return num1 - num2
    elif operator == "*":
        return num1 * num2
    elif operator == "/":
        return num1 / num2
    else:
        return "invalid operator"


num1 = 45 # float(input("Enter a: "))
operator = "/" # input("Enter operator: ")
num2 = 3 # float(input("Enter b: "))
print("Answer is: " + str(cal(num1,num2,operator)))

#Dictionary
monthConversion = {
    "Jan":"January",
    "Feb":"February",
    "Mar":"March"
}

print(monthConversion["Feb"])
print(monthConversion.get("Mar"))
print(monthConversion.get("BlaBla", "Not Available"))


#While
i=1
while i<=5:
    print(i)
    i+=1
print("Done with while")

#GussingGame
secretWord = "Giraffe"
guess = "Giraffe" #guess = ""
guessCount = 1
guessLimit = 5
outOfGuess = False
while guess != secretWord and not outOfGuess:
    if(guessCount < guessLimit):
        guess = input("Enter guess: ")
        guessCount+=1
    else:
        outOfGuess = True
if(not outOfGuess):
    print('you win')
else:
    print('you Lose')



#For Loop

for letter in "each Letter here":
    print(letter)

friends = ["Ram", "Shyam", "Ghansham"]
for name in friends:
    print(name)

friends = ["Ram", "Shyam", "Ghansham"]
for name in range(len(friends)):
    print(name)
    print(friends[name])


#Exponent
def raiseToPower(baseNum, powNum):
    result = 1
    for index in range(powNum):
        result = result * baseNum
    return result
print(raiseToPower(2,4))
print(2**4)

#2D Lists & Nested Loops

numGrid =[
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [0]
]

for row in numGrid:
    print (row)
    for col in row:
        print(col)

print(numGrid[0][0])
print(numGrid[3][0])

#Building a Translator

def translate(phrase):
    translation = ""
    for letter in phrase:
        if letter in "AEIOUaeiou":
            translation= translation + "g"
        else:
            translation = translation + letter
    return translation

print (translate("This is my phrase to be translated"))
#print (translate(input("Enter the phrase : ")))

#Comments
'''
Write
As many comnet
as you want
Noramlly # is prefferesd
'''

# Try an except

try:
   # value = 10/0
    #num = int(input("Enter a number " ))
    num = 10
    print(num)
except ZeroDivisionError:
    print("Zero Division ")

except ValueError:
    print("Invalid input")


#Reading Files
# r read
# r+ read & write
# a append
employeeFile = open("employee.txt","r")
if (employeeFile.readable()):
    print(employeeFile.read())
#    print(employeeFile.readlines())
#    print(employeeFile.readline())
print(employeeFile.readable())
employeeFile.close()


#Edit the file
employeeFileName = "employee.txt"
employeeFile = open(employeeFileName,"a")
#employeeFile.write("\nToby - Human Resources")
if (employeeFile.readable()):
    print(employeeFile.read())
#    print(employeeFile.readlines())
#    print(employeeFile.readline())
print(employeeFile.readable())
employeeFile.close()

#Module &
# https://docs.python.org/3/py-modindex.html

fileExt = getFileExt(employeeFileName)
print(fileExt)
rand = usefulTools.rollDice(10)
#print("ccc" + str(rand))

#python -m pip install --upgrade pip
#pip install python-docx
# installed under C:\Users\nv\AppData\Local\Programs\Python\Python38-32\Lib\site-packages\docx
#pip uninstall python-docx


#Classes & Objects

student1 = Student("Jim","Business",3.1,False)
student1.info()


print("End")



