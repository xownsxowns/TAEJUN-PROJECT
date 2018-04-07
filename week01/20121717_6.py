## Lab Quiz #6
## ?JA9s6


text = open('world_population_area.txt', 'rt')
name = []
population = []
area = []
density = []
for line in text:
    a = line.split('\t')
    a[0] = a[0].replace('"','')
    a[1] = a[1].replace(' ','')
    a[2] = a[2].replace(' \n','')
    den = int(a[1])/int(a[2])
    name.append(a[0])
    population.append(a[1])
    area.append(a[2])
    density.append(den)



while True:
    instruction = input("Enter a instruction (p: print all info, d: print all density, e: exit): ")
    if instruction == 'p':
        print('---------------------------------')
        print('|Country\t|Population\t|Area\t|')
        print('---------------------------------')
        for i in range(len(name)):
            print('|{0}\t|\t{1}|\t{2}|'.format(name[i], population[i], area[i]))

    elif instruction == 'd':
        print('-------------------------')
        print('|Country\t|Density\t|')
        print('-------------------------')
        for i in range(len(name)):
            print('|{0}\t|\t{1}|'.format(name[i], density[i]))

    elif instruction == 'e':
        break