import random

a = random.randint(0, 100)
b = random.randint(0, 100)

if a < b:
    print("Toinen numero on suurempi")
elif b < a:
    print("Ensimmäinen numero on suurempi")
else:
    print("Yhtäsuuret")