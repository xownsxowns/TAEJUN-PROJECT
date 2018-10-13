
def add(a, b):
    print(a + b)

def mul(a, b):
    print(a * b)

def indexing(a, b):
    print(a[b])

def concate(a, b):
    print(a,b)

def div(a,b):
    c = a.count(b)
    print(c)

def sub(a,b):
    a = list(a)
    for i in range(len(b)):
        if b[i] in a:
            c = a.index(b[i])
            a.pop(c)
        else:
            continue
    print("".join(a))
