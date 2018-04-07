
while True:
    instruction = input("Enter a instruction (p: print all info, d: print all density, e: exit): ")
    if instruction == 'p':
        f = open('world_population_area.txt', 'rt')
        print("-"*64)
        print("|{0:^20s}|{1:^20s}|{2:^20s}|".format('Country', 'Population', 'Area'))
        print("-"*64)
        for line in f:
            info = line.split('\t')
            print("|{0:<20s}|{1:>20,d}|{2:>20,d}|".format(info[0].strip('\"'), int(info[1].strip()), int(info[2].strip())))
        print("-"*64)
        f.close()
    elif instruction == 'd':
        f = open('world_population_area.txt', 'rt')
        density = dict()
        print("-" * 38)
        print("|{0:<20s}|{1:>15s}|".format('Country', 'Density'))
        print("-" * 38)
        for line in f:
            info = line.split('\t')
            index = info[0].strip('\"')
            density[index] = float(info[1].strip()) / float(info[2].strip())
            print("|{0:<20s}|{1:>15.2f}|".format(index, density[index]))
        print("-" * 38)
        f.close()
    elif instruction == 'e':
        print("Program is done.")
        break
    else:
        print("Wrong instruction... p or d or e!")
