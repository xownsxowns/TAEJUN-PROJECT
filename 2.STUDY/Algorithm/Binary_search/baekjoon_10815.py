from bisect import bisect_left, bisect_right

N = int(input())
cards = sorted(list(map(int, input().split())))
M = int(input())
qry = list(map(int, input().split()))

ans = []
for q in qry:
    l = bisect_left(cards, q)
    r = bisect_right(cards, q)
    if r - l > 0:
        ans.append(1)
    else:
        ans.append(0)
    
print(*ans)