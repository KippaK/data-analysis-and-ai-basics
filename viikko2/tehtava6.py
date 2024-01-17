class Fraction:
    def __init__(self, nominator, denominator):
        self.nominator = nominator
        self.denominator = denominator

    def print(self):
        print(self.nominator, "/", self.denominator)
    
    def simplification(self):
        while (True):
            x = True
            for i in range(int(max(self.nominator, self.denominator)), int(2), int(-1)):
                if (self.nominator % i == 0) and (self.denominator % i == 0):
                    self.nominator = int(self.nominator / i)
                    self.denominator = int(self.denominator / i)
                    x = False
            
            if x:
                break
    


num = Fraction(10, 50)
num.print()
num.simplification()
num.print()