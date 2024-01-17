a = int(input("Syötä ensimmäinen numero: "))
b = int(input("Syötä toinen numero: "))

if a < b:
    print("Toinen numero on suurempi")
elif b < a:
    print("Ensimmäinen numero on suurempi")
else:
    print("Yhtäsuuret")
