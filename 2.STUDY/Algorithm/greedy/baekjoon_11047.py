N, K = map(int, input().split())

coins = [int(input()) for _ in range(N)]
coins.reverse()
ans = 0

for coin in coins:
    ans += K // coin
    K %= coin

print(ans)

# # My answer
# coin_list = [int(input()) for _ in range(N)]
# coin_list.reverse()
    
# total_n = 0
# for coin in coin_list:
#     n = K // coin
#     if n > 0:
#         total_n = total_n + n
#         K = K - (coin*n)
#         continue
    
#     if K == 0:
#         break

# print(total_n)
    
    