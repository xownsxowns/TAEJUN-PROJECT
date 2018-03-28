

answer_number = input("Enter the answer number: ")
guess_number = input("Guess the number: ")

while True:
    if guess_number > answer_number:
        guess_number = input("It's lower. Guess another number: ")
    elif guess_number < answer_number:
        guess_number = input("It's higher. Guess another number: ")
    else:
        break

print("That's good")
