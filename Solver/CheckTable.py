class CheckTable:
    def __init__(self, table):
        self.NumTable = table

    def checkValidity(self):
        retVal = True
        for i in range(9):
            retVal &= self.checkColumn(i)
            retVal &= self.checkRow(i)

        for i in range(3):
            for j in range(3):
                retVal &= self.checkBlock(i, j)

        return retVal

    def checkRow(self, row):
        nums = [False for i in range(9)]

        for i in range(9):
            nums[self.NumTable[row][i] - 1] = True  # -1, mert 0-8ig van az index, 1-9ig a szam

        retVal = True
        for b in nums:
            retVal &= b

        return retVal

    def checkColumn(self, column):
        nums = [False for i in range(9)]

        for i in range(9):
            nums[self.NumTable[i][column] - 1] = True  # -1, mert 0-8ig van az index, 1-9ig a szam

        retVal = True
        for b in nums:
            retVal &= b

        return retVal

    def checkBlock(self, blockRow, blockColumn):
    # blockRow es blockColumn 0 es 2 kozott

        blockRow *= 3
        blockColumn *= 3

        nums = [False for i in range(9)]

        for i in range(3):
            for j in range(3):
                ni = i + blockRow
                nj = j + blockColumn
                nums[self.NumTable[ni][nj] - 1] = True

        retVal = True
        for b in nums:
            retVal &= b

        return retVal