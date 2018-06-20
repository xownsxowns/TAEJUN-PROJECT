def permutations(word):
    result = []
    if len(word) == 0:
        result.append(word)
        return result
    else:
        for i in range(len(word)):
            shorter = word[:i] + word[i+1:]
            shorterPermutations = permutations(shorter)
            for string in shorterPermutations:
                result.append(word[i] + string)
        return result

for string in permutations("eat"):
    print(string)