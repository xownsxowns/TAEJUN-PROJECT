d = dict()
for _ in range(int(input())):
    book = input()
    if book in d:
        d[book] += 1
    else:
        d[book] = 1

m = max(d.values())
candi = []
for k, v in d.items():
    if v == m:
        candi.append(k)
print(sorted(candi)[0])

## My Answer
# book_list = dict()
# n = int(input())

# for _ in range(n):
#     book_name = input()
#     if book_name in book_list.keys():
#         book_list[book_name] += 1
#     else:
#         book_list[book_name] = 1
        
# book_list = dict(sorted(book_list.items()))
# print(max(book_list, key=book_list.get))