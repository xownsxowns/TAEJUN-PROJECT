N, L = map(int, input().split())
coord = [False] * 1001

for i in map(int, input().split()):
    coord[i] = True

n_tape = 0
x = 0
while x < 1001:
    if coord[x]:
        n_tape += 1
        x += L
    else:
        x += 1

print(n_tape)