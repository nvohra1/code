import string
alphabet = " " + string.ascii_lowercase

positionToLetters = {i:alphabet[i] for i in range(0, len(alphabet))}

letters = {alphabet[i]:i for i in range(0,len(alphabet))}
nPosition = letters['n']
print(positionToLetters[nPosition])

toEncodeShiftFwd = 3
message = "hi my name is caesar"

def encode(message, toEncodeShiftFwd):
    xmsg=''
    encodedMsg=xmsg.join(positionToLetters[ (letters[letter]+toEncodeShiftFwd) if (letters[letter]+toEncodeShiftFwd) < 27 else (letters[letter]+toEncodeShiftFwd) - 27      ] for letter in message)
    return encodedMsg

t = encode(message, toEncodeShiftFwd)
print(t)