for _ in range(int(input())):
    stk = []
    isVPS = True
    for i in input():
        if i == '(':
            stk.append(i)
        else:
            if stk:
                stk.pop()
            else:
                isVPS = False
                break

    if stk:
        isVPS = False

    print('YES' if isVPS else 'NO')