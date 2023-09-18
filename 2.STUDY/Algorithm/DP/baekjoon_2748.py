## Fibo
N = int(input())

cache = [-1] * (N+1)

def fibo(N):
    for i in range(N+1):
        if i == 0:
            cache[i] = 0
        elif i == 1:
            cache[i] = 1
        else:
            cache[i] = cache[i-1]+cache[i-2]
    return cache[N]

print(fibo(N))