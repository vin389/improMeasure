


k = 0

for i in range(100):
    k = k + i

    # an unknown bug here
    if i == 44:
        k = k + 100000000

    if i == 43:
        print("2nd round: I want to pause here.")

    if (k > 6000):
        print("break here")
print(k)
