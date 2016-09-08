from Solver.SolvingTable import SolvingTable

table = [[-1 for i in range(9)] for j in range(9)]

file = "D:\Dokumentumok\Visual Studio 2010\Projects\cs\Sudoku\sudoku5.csv"
filestream = open(file)

i = 0

for line in filestream:
    strs = line[:-1].split(';')
    for j in range(9):
        if strs[j] == "":
            table[i][j] = -1
        else:
            table[i][j] = int(strs[j])

    i += 1
    if i >= 9:
        break

st = SolvingTable(table)
st.write()
st.SolveBackTrack()
st.write()
