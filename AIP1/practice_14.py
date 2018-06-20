def print_post_order(in_order, pre_order):
    root = in_order.index(pre_order[0])
    if root != 0:
        print_post_order(in_order[:root], pre_order[1:root+1])
    if root != len(in_order)-1:
        print_post_order(in_order[root+1:],pre_order[root+1:])
    print(pre_order[0], end=' ')

def main():
    in_order = [4,2,5,1,3,6]
    pre_order = [1,2,4,5,3,6]
    print("Postorder traversal: ", end='')
    print_post_order(in_order, pre_order)

if __name__ == '__main__':
    main()
