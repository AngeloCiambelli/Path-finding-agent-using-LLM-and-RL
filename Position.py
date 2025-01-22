

class Position:
    x: int
    y: int
    nature: str

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Position({self.x}, {self.y})"

    def get_nature_according_to_environment(self, env) -> str:
        is_bottom_left_corner = self.x == 0 and self.y == 0
        is_bottom_right_corner = self.x == env.size - 1 and self.y == 0
        is_top_left_corner = self.x == 0 and self.y == env.size - 1
        is_top_right_corner = self.x == env.size - 1 and self.y == env.size - 1

        is_left_border = self.x == 0 and 0 < self.y < env.size - 1
        is_bottom_border = self.y == 0 < self.x < env.size - 1
        is_right_border = self.x == env.size - 1 and 0 < self.y < env.size - 1
        is_top_border = self.y == env.size - 1 and 0 < self.x < env.size - 1

        if is_bottom_left_corner:
            nature = 'bottom left corner'
        elif is_bottom_right_corner:
            nature = 'bottom right corner'
        elif is_top_left_corner:
            nature = 'top left corner'
        elif is_top_right_corner:
            nature = 'top right corner'

        elif is_left_border:
            nature = 'left border'
        elif is_right_border:
            nature = 'right border'
        elif is_bottom_border:
            nature = 'bottom border'
        elif is_top_border:
            nature = 'top border'

        else:
            nature = 'free'

        return nature
