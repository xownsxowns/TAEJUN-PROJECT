N, K = map(int, input().split())

######### TOP-DOWN ###########
# cache = [[0] * 1001 for _ in range(1001)]

# def bino(N, K):
#     if (K == 0) or (N == K):
#         cache[N][K] = 1
#     else:
#         cache[N][K] = bino(N-1, K-1) + bino(N-1, K)
#         cache[N][K] %= 10007
    
#     return cache[N][K]

# print(bino(N, K))

########## BOTTOM-UP ###############
cache = [[0] * 1001 for _ in range(1001)]
for i in range(1001):
    cache[i][0] = cache[i][i] = 1
    for j in range(1, i):
        cache[i][j] = cache[i-1][j-1] + cache[i-1][j]
        cache[i][j] %= 10007

print(cache[N][K])