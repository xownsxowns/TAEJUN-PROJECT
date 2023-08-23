import heapq as hq
import sys

input = sys.stdin.readline
pq = []
for _ in range(int(input())):
    x = int(input())
    if x:
        hq.heappush(pq, (abs(x), x))
    else:
        print(hq.heappop(pq)[1] if pq else 0) # 원본을 원하니깐 tuple에서 index [1]
