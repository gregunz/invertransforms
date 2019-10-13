from invertransforms.lambd import Lambda


class Identity(Lambda):
    def __init__(self):
        super().__init__(lambd=lambda x: x)

    def invert(self):
        return Identity()
