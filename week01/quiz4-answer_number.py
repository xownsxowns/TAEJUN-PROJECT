'''
We make a program guess the answer number.
But we already know the answer number, thus you think that this program is
just checking wheter this program can make a proper decision depending on
the guess number.
If the guess number is less than answer, we print out "It's higher than it. Guess another number: ", and input another number. Otherwise, if the guess number is greater than the answer, print out "It's lower than it. Guess another number: " and input another number.
We continue to do this job until the guess number is equal to the answer number.
We can use while statement to express the loop for this quiz.
We're ending this program with the string "That's good!" when we answer correct number.
'''

def is_same(answer, guess):
    if answer == guess:
        return "Win"
    elif answer > guess:
        return "Low"
    else:
        return "High"


answer = int(input("Enter the answer number: "))
guess = int(input("Guess the number: "))
higher_or_lower = is_same(answer, guess)

while higher_or_lower != "Win":
    if higher_or_lower == "Low":
        guess = int(input("It's higher. Guess another number : "))
    else:
        guess = int(input("It's lower. Guess another number : "))
    higher_or_lower = is_same(answer, guess)

print("That's good!")
