#Wczytywanie danych z pliku - zadania
#%config IPCompleter.greedy=True

with open('D:\MachineLearningPandaIT\Materials\male.txt' , 'r') as fileMale:
    fileMale = fileMale.read().split(',')
   
with open('D:\MachineLearningPandaIT\Materials\\female.txt' , 'r') as fileFemale:
    fileFemale = fileFemale.read().split(',')

#to define letter below
letter = "p"
#to define letter above

def countMale(file,letter):
    countM = 0
    for name in file:
        if name.strip().lower().startswith(letter):
            countM += 1
    return countM

print("result for Male is " , countMale(fileMale,letter))


print("-----")
#for females here
def countFemale(file,letter):
    countF = 0
    for name in file:
        if name.strip().lower().startswith(letter):
            countF += 1
    return countF

print("result for Female is " , countFemale(fileFemale,letter))

print("-----")

countM = countMale(fileMale,letter)
countF = countFemale(fileFemale,letter)

def compareGenders (countM,countF):
    if countM > countF:
        print("more Male names")
    elif countM < countF:
        print("more Female names")
    elif countM == countF:
        print("number of names is equal")

print(compareGenders(countM,countF))

#unit test za pomoca assert i try-catch
print("***unit test start***")
try:
    assert countMale(fileMale,'p') == 4
except:
    print("Exception found")
finally:
    print("***unit test end***")



print("-----")
print(sorted(fileMale))
print("-----")
print(sorted(fileFemale))