

class Position:
    x: int
    y: int
    nature: str

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Position({self.x}, {self.y})"

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)
