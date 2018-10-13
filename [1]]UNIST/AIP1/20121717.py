## 20121717

def binary_search(a_list, item):
    first = 0
    last = len(a_list) - 1
    found = False
    while first <= last and not found:
        midpoint = (first + last) // 2
        if a_list[midpoint] == item:
            found = True
        else:
            if item < a_list[midpoint]:
                last = midpoint - 1
            else:
                first = midpoint + 1
    return found

def bubble_sort(a_list):
    for j in range(len(a_list)-1):
        for i in range(len(a_list)-1):
            current_value = a_list[i]
            if a_list[i] > a_list[i+1]:
                a_list[i] = a_list[i+1]
                a_list[i+1] = current_value

def selection_sort(a_list):
    list = a_list
    a_list = []
    t = 0
    for i in range(len(list) - 1):
        a = min(list)
        b = list.pop(list.index(a))
        a_list.insert(t,b)
        t= t+1
    return a_list

def insertion_sort(a_list):
    for i in range(1, len(a_list)):
        current_value = a_list[i]
        pos = i
        while pos > 0 and a_list[pos - 1] > current_value:
            a_list[pos] = a_list[pos - 1]
            pos = pos - 1
        a_list[pos] = current_value

def main():
    count = int(input("Enter the number of numbers: "))
    num_list = []
    for i in range(0, count):
        number = int(input("Enter the number: "))
        num_list.append(number)
    print("The number list is: ", num_list)

    while True:
        type = input("Enter the types of sort: ")
        if type == 'bubble':
            bubble_sort(num_list)
            break
        elif type == 'selection':
            selection_sort(num_list)
            break
        elif type == 'insertion':
            insertion_sort(num_list)
            break
        else:
            continue

    print("The sorted number list is: ", num_list)
    target = int(input("Enter the target number to find: "))
    result = binary_search(num_list, target)
    if result == True:
        print("We found!")
    else:
        print("We cannot found!")

if __name__ == '__main__':
    main()