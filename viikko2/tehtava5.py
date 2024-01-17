import random

ints1 = []
ints2 = []
correct_ans = []

for i in range(0, 5):
    ints1.append(random.randint(0, 10))
    ints2.append(random.randint(0, 10))
    correct_ans.append(ints1[i] * ints2[i])

ans = []
correct_count = 0

print("Laske kertolaskut: ")
for i in range(0, 5):
    s = str(ints1[i]) + " * " + str(ints2[i]) + " = "
    ans.append(int(input(s)))

for i in range(0, 5):
    if correct_ans[i] == ans[i]:
        print("Oikein!", ints1[i], "*", ints2[i], "=", correct_ans[i])
    else:
        print("Väärin. Oikea vastaus on:", ints1[i], "*", ints2[i], "=", correct_ans[i])
