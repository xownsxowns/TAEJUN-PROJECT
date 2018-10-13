import numpy as np

def get_matrix(x):
    mat = []
    [n, m] = [int(x) for x in x.strip().split(" ")]
    for i in range(n):
        row = [int(x) for x in input().strip().split(" ")]
        mat.append(row)
    return np.array(mat)

def main():
    x = input("Enter the matrix A: ")
    A = get_matrix(x)
    x = input("Enter the matrix B: ")
    B = get_matrix(x)
    x = input("Enter the matrix C: ")
    C = get_matrix(x)

    print(A+B)
    print(A-B)
    print(A*B)
    print(np.matmul(A,C))
    print(np.transpose(A))
    print(np.count_nonzero(A > 2))
    print(sum(sum(A))+sum(sum(B)))
    print(B*10)
    print(A**3)

    float_A = np.array(A, dtype=float)
    print(float_A)


if __name__ == "__main__":
    main()