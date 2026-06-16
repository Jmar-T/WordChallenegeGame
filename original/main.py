import random
acceptedChars = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
TheFile = open("original/Dictionary", "r")
File2 = open("NewDict", "w")

lengthOfWordle = int(input("What length of the wordle would you like? :"))
valid_words = []
adict = {}
count = 0
for line in TheFile:
    line = line.strip()
    line = line.split(" ")
    definition = line[2:]
    definition = " ".join(definition)
    validWord = True
    for char in line[0]:
        if char not in acceptedChars:
            validWord = False
    if len(line[0]) == lengthOfWordle and validWord and line[1] == "":
        adict[line[0]] = definition
        valid_words.append(line[0].lower())
for key, value in adict.items():
    print(key + "/t" + value, file=File2)
    count += 1

File2.close()
TheFile.close()
File2 = open("NewDict", "r")

n = random.randint(0, count)
counter = File2.readlines()
theLine = counter[n]

theLine = theLine.split("/t")
theLine[0] = theLine[0].lower()

theString = ""
wordsNotInList = []
wordleComplete = False
rounds = 5
while rounds > 0:
    wordsWrongPlace = []
    wordAttempt = input("")
    if len(wordAttempt) != lengthOfWordle:
        print("Length of word is incorrect, try again!")
        wordAttempt = input("")
    if wordAttempt not in valid_words:
        print(wordAttempt + " is not in dictionary, try again")
        wordAttempt = input("")
    if wordAttempt == theLine[0]:
        print("Congratulations! " + theLine[0] + ": " + theLine[1])
        wordleComplete = True
        break
    else:
        for x in range(lengthOfWordle):
            if theLine[0][x] == wordAttempt[x]:
                theString += wordAttempt[x]
            else:
                theString += "-"
                if wordAttempt[x] in theLine[0]:
                    if wordAttempt[x] not in wordsWrongPlace:
                        wordsWrongPlace.append(wordAttempt[x])
                else:
                    if wordAttempt[x] not in wordsNotInList:
                        wordsNotInList.append(wordAttempt[x])

    print(theString + "\n The following letters are not in the word " + "-".join(wordsNotInList) +
          "\n The following letters are in the word at a different location " + "-".join(wordsWrongPlace))
    theString = ""
    rounds -= 1
if not wordleComplete:
    print("FAILURE TO COMPLETE. The word was " + theLine[0] + ": " + theLine[1])