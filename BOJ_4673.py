originalNumber = set(range(1,10001))
generatorEx = set()


for n in range(1,10001) :
    for i in str(n):
        n += int(i)
    generatorEx.add(n)

selfNumber = sorted(originalNumber - generatorEx)

for n in selfNumber:
    print(n)


