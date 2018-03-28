info = {}

while True:
    instruction = input("Enter a instruction (i: insertion, d: display, e: exit): ")
    if instruction == 'd':
        for student in info.keys():
            print("{0}'s average subject score is {1}".format(student, sum(info[student_name]['subject score'])/len(info[student_name]['subject score'])))

    elif instruction == 'i':
        student_name = str(input("Enter the student name:"))
        subject_name = input("Enter the subject name:")
        subject_score = int(input("Enter the subject score:"))
        if student_name in info:
            info[student_name][subject_name]=subject_score
        else:
            info[student_name] = {'subject name': [subject_name], 'subject score': [subject_score]}


    elif instruction == 'e':
        print("Program is done")
        break

    else:
        print("Wrong instruction... i or d or e!")

print(info)