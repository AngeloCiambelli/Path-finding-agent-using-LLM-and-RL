

class Position:
    x: int
    y: int
    type: str

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Position({self.x}, {self.y})"

    def get_type_according_to_environment(self, env) -> str:
        is_bottom_left_corner = self.x == 0 and self.y == 0
        is_bottom_right_corner = self.x == env.size - 1 and self.y == 0
        is_top_left_corner = self.x == 0 and self.y == env.size - 1
        is_top_right_corner = self.x == env.size - 1 and self.y == env.size - 1

        is_left_border = self.x == 0 and 0 < self.y < env.size - 1
        is_bottom_border = self.y == 0 < self.x < env.size - 1
        is_right_border = self.x == env.size - 1 and 0 < self.y < env.size - 1
        is_top_border = self.y == env.size - 1 and 0 < self.x < env.size - 1

        if is_bottom_left_corner or is_bottom_right_corner or is_top_left_corner or is_top_right_corner:
            type = 'corner'

        elif is_left_border or is_right_border or is_bottom_border or is_top_border:
            type = 'border'

        else:
            type = 'free'

        return type
