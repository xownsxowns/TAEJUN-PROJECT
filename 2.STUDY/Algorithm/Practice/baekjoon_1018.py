# 체스판 다시 칠하기
N, M = map(int, input().split())
board = [input() for _ in range(N)]
ans = 64 # 최대값을 넣어두고 최소값으로 