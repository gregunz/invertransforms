from . import Lambda


class Identity(Lambda):
    def __init__(self):
        super().__init__(
            lambd=lambda x: x,
            lambd_inv=lambda x: x,
        )

    def __repr__(self):
        return 'Lambda x: x'
