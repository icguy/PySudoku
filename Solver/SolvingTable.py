from Tile import *
from CheckTable import *

class SolvingTable:
    # table is list of lists of integers, 9x9
    def __init__(self, table):
        self.tileTable = [[Tile(table[i][j]) for j in range(9)] for i in range(9)]

    def SolveBackTrack(self):
        st = self.Backtrk()
        return st.TestSolved()

    def Backtrk(self):
        if self.TestComplete():
            return self

        for i in range(9):
            for j in range(9):
                if self.tileTable[i][j].Possibilities is None:
                    self.ApplyTileToSurrounding(i, j)

                    if self.TestComplete():
                        return self

        # guess
        r , c= -1, -1
        min = 10
        for i in range(9):
            gotval = False
            for j in range(9):

                if self.tileTable[i][j].Possibilities is None:
                    continue

                if self.tileTable[i][j].NumPossible == 2:
                    r = i
                    c = j
                    gotval = True
                    min = 2
                    break

                elif self.tileTable[i][j].NumPossible < min:
                    r = i
                    c = j
                    min = self.tileTable[i][j].NumPossible

            if gotval:
                break

        sts = [None for i in range(min)]
        print "guessing: " + str(min)

        nextidx = 0
        for i in range(9):
            if self.tileTable[r][c].Possibilities[i]:
                newtable = self.copytable2(self.tileTable)
                # if newtable[r][c].Possibilities is None:
                #     assert False
                newtable[r][c] = i + 1
                sts[nextidx] = SolvingTable(newtable)
                nextidx += 1

        for st in sts:
            if st.SolveBackTrack():
                self.tileTable = st.tileTable
                break

        return self

    def write(self):
        from pprint import pprint
        table = [[t.Value if t.Value != -1 else 0 for t in row] for row in self.tileTable]
        pprint(table)

    def ApplyTileToSurrounding(self, row, column):

        if (self.tileTable[row][column]).Value == -1:
            return

        num = self.tileTable[row][column].Value
        for i in range(9):
            if i != row:
                if self.tileTable[i][column].RemovePossibility(num):
                    self.ApplyTileToSurrounding(i, column)

            if i != column:
                if self.tileTable[row][i].RemovePossibility(num):
                    self.ApplyTileToSurrounding(row, i)

        bi = row - row % 3
        bj = column - column % 3
        for i in range(bi, bi + 3):
            for j in range(bj, bj + 3):
                if bi == row and bj == column:
                    continue
                if self.tileTable[i][j].RemovePossibility(num):
                    self.ApplyTileToSurrounding(i, j)

    def GuessNextTileToSolve(self):
        minNum = 10
        gi = -1
        gj = -1
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    numPoss = self.tileTable[i][j].NumPossible
                    if minNum > numPoss > 0:
                        minNum = numPoss
                        gi = i
                        gj = j
        return gi, gj

    def TestSolved(self):
        # True ha meg van oldva
        if not self.TestComplete():
            return False
        newtable = [[0 for i in range(9)] for j in range(9)]

        for i in range(9):
            for j in range(9):
                newtable[i][j] = self.tileTable[i][j].Value

        ct = CheckTable(newtable)
        return ct.checkValidity()

    def TestComplete(self):
        # True ha nincs -1 a tombben
        for row in self.tileTable:
            for t in row:
                if t.Value == -1:
                    return False
        return True

    def copytable(self, table):
        return [row[:] for row in table]

    def copytable2(self, table):
        return [[t.Value for t in row] for row in table]