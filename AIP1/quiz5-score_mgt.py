records = {}
dict1 = {}

while True:
    inst = input("Enter a instruction (i: insertion, d: display, e: exit): ")
    if inst == 'i':
        student_name = input("Enter the student name: ")
        sub_name = input("Enter the subject name: ")
        sub_score = int(input("Enter the subject score: "))
        if student_name not in records:
            records[student_name] = dict()
        records[student_name][sub_name] = sub_score
    elif inst == 'd':
        for key, value in records.items():
            sum_v = 0
            for v in value.values():
                sum_v += v
            print(key, "'s average subject score is ", sum_v / len(value), sep='')
    elif inst == 'e':
        print("Program is done")
        break
    else:
        print("Wrong instruction... i or d or e!")
