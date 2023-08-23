from numpy import *


class simplexmethod:
    def __init__(self, objective):
        self.objective = [1] + objective
        self.rows = []
        self.cons = []

    def add_constraint(self, expression, RHS):
        self.rows.append([0] + expression)
        self.cons.append(RHS)

    # pivot column을 찾아서 entering basic variable을 return
    def pivot_column(self):
        low = 0
        entering_x = 0
        for i in range(1, len(self.objective) - 1):
            if self.objective[i] < low:
                low = self.objective[i]
                entering_x = i
        if entering_x == 0:
            return -1
        return entering_x

    # pivot row를 찾아서 Departing variable을 return
    def pivot_row(self, col):
        rhs = [self.rows[i][-1] for i in range(len(self.rows))]
        a = [self.rows[i][col] for i in range(len(self.rows))]
        ratio = []
        for i in range(len(rhs)):
            if a[i] == 0:
                ratio.append(99999999 * abs(max(rhs)))
                continue
            ratio.append(rhs[i] / a[i]) # smallest ratio test
        return argmin(ratio)

    def show(self):
        print('\n', matrix([self.objective] + self.rows))

    def pivot(self, row, col):
        e = self.rows[row][col]
        self.rows[row] /= e
        for r in range(len(self.rows)):
            if r == row:
                continue
            self.rows[r] = self.rows[r] - self.rows[r][col] * self.rows[row]
        self.objective = self.objective - self.objective[col] * self.rows[row]

    def check(self):
        if min(self.objective[1:-1]) >= 0: # check there are only non-negative values
            return 1
        return 0

    def solve(self):

        # build full tableau
        for i in range(len(self.rows)):
            self.objective += [0]
            identify = [0 for r in range(len(self.rows))]
            identify[i] = 1
            self.rows[i] += identify + [self.cons[i]]
            self.rows[i] = array(self.rows[i], dtype=float)
        self.objective = array(self.objective + [0], dtype=float)

        # solve
        self.show()
        while not self.check():
            c = self.pivot_column()
            r = self.pivot_row(c)
            self.pivot(r, c)
            print('\npivot column: %s\npivot row: %s' % (c + 1, r + 2))
            self.show()


if __name__ == '__main__':
    """
    max Z = 23 + 22x + [-25 -9 56 2][yi] - 21z
    st
    x <= 230801
    z <= 97
    -10x + 7z <= 5
    x,y,z >= 0
    """

    chicken = simplexmethod([-22,21])
    chicken.add_constraint([1, 0], 230801)
    chicken.add_constraint([0, 1], 97)
    chicken.add_constraint([-10, 7], 0)
    chicken.solve()