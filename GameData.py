from random import randint

"""self.field = [
            [Cell(4), Cell(8), Cell(4), Cell(2)],
            [Cell(2), Cell(16), Cell(32), Cell(4)],
            [Cell(8), Cell(32), Cell(16), Cell(2)],
            [Cell(32), Cell(128), Cell(32), Cell(16)]
]"""


class Cell:
    def __init__(self, number=0):
        self.number = number
        self.str_number = str(number)

    def get_number(self):
        return self.number

    def get_str_number(self):
        return self.str_number

    def change_number(self, number):
        self.number = number
        self.str_number = str(number)

    def __add__(self, other):
        return Cell(self.get_number() + other.get_number())

    def __eq__(self, other):
        return self.get_number() == other.get_number()

    def __repr__(self):
        return self.str_number


class GameData:
    progress = 0
    crt_progress = 0
    moves = 0

    def __init__(self):
        self.refresh()

    def refresh(self):
        self.progress = 0
        self.crt_progress = 0
        self.moves = 0
        self.field = [[Cell() for _ in range(4)] for _ in range(4)]
        self.rand_cell()

    def get_field(self):
        return self.field

    def get_number_field(self):
        return [[cell.get_number() for cell in row] for row in self.field]

    def rand_cell(self):
        empty_cell = Cell()
        empty_cells = []
        for i in range(4):
            for j in range(4):
                if self.field[i][j] == empty_cell:
                    empty_cells.append((i, j))

        if not empty_cells:
            return

        num = 4 if randint(1, 11) == 11 else 2
        i, j = empty_cells[randint(0, len(empty_cells) - 1)]
        self.field[i][j] = Cell(num)

    def move(self, num, for_progress=True):
        if not for_progress:
            field = self.field.copy()

        check = False
        if num == 0:
            self.field = list(map(list, list(zip(*self.field))[::-1]))
            check = self.try_move(for_progress)
            self.field = list(map(list, zip(*self.field[::-1])))
        elif num == 1:
            list(map(lambda x: x.reverse(), self.field))
            check = self.try_move(for_progress)
            list(map(lambda x: x.reverse(), self.field))
        elif num == 2:
            self.field = list(map(list, zip(*self.field[::-1])))
            check = self.try_move(for_progress)
            self.field = list(map(list, list(zip(*self.field))[::-1]))
        elif num == 3:
            check = self.try_move(for_progress)

        self.progress += self.crt_progress
        if not for_progress:
            self.field = field
        elif check:
            self.moves += 1

        return check

    def try_move(self, for_progress):
        # проверяет возможен ли ход
        check = False

        for num, row in enumerate(self.field):
            # without_0 = list(filter(lambda cell: cell.get_number() != 0, row))
            cell_None = Cell()
            check_None = False
            without_0 = []
            for cell in row:
                if cell == cell_None:
                    check_None = True
                    continue
                without_0.append(cell)
                if check_None:
                    check = True

            i = 1
            while i < len(without_0):
                if without_0[i - 1] == without_0[i]:
                    without_0[i-1:i+1] = [without_0[i - 1] + without_0[i]]
                    if for_progress:
                        self.crt_progress += without_0[i-1].get_number()
                    check = True
                i += 1

            for _ in range(4 - len(without_0)):
                without_0.append(Cell())

            self.field[num] = without_0

        return check

    def check_GameOver(self):
        for row in self.field:
            if Cell() in row:
                return False

        for i in range(4):
            if self.move(i, False):
                return False

        return True

    def __repr__(self):
        return '\n'.join(
            list(map(
                    lambda row: ' '.join(list(map(str, row))),
                    self.field
                )
            )
        )
