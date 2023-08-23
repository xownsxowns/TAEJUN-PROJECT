from itertools import combinations

heights = [int(input()) for _ in range(9)]
for combi in combinations(heights):
    if sum(combi) == 100:
        for height in sorted(combi):
            print(height)
        break

# # My answer
# hei_list = []
# for _ in range(9):
#     hei = int(input())
#     hei_list.append(hei)

# for i in combinations(hei_list, 7):
#     if sum(i) == 100:
#         ans = i
#         break

# ans = sorted(ans)
# for i in range(len(ans)):
#     print(ans[i])
