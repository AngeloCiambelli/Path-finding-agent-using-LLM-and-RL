from Position import Position


class Agent:
    position: Position

    def __init__(self, initial_position: Position):
        self.position = initial_position
