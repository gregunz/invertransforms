from invertransforms.lambd import Lambda


class Identity(Lambda):
    def __init__(self):
        super().__init__(
            lambd=lambda x: x,
            lambd_inv=lambda x: x,  # not used because _invert is overridden (for __repr__)
        )

    def _invert(self):
        return Identity()
