class Tile:
    # int
    # self.Value
    # # -1 ha nincs meg
    #
    # bool[]
    # self.Possibilities
    # # None ha megvan a val
    # # True az erteke, ha az a szam meg lehet
    #
    # int
    # self.NumPossible

    def __init__(self, value):
        self.Value = value
        if self.Value == -1:
            self.Possibilities = [False for i in range(9)]
            for i in range(9):
                self.Possibilities[i] = True
            self.NumPossible = 9
        else:
            self.Possibilities = None
            self.NumPossible = 1

    def RemovePossibility(self, num):
        # return True, ha megvan a self.Value
        num -= 1
        if self.NumPossible == 1:
            return False

        if num > len(self.Possibilities) or num < 0:
            assert False

        if not self.Possibilities[num]:
            return False

        self.Possibilities[num] = False
        self.NumPossible -= 1

        if self.NumPossible == 1:
            for i in range(9):
                if self.Possibilities[i]:
                    self.Value = i + 1
                    break

            self.Possibilities = None
            return True
        return False

    def AddValue(self, value):
        if self.Value != -1:
            assert False
        self.Value = value
        self.NumPossible = 1
        self.Possibilities = None

    def __repr__(self):
        return "(" + str(0 if self.Value == -1 else self.Value) + "p" + str(self.NumPossible) + ")"