import enum


class CallableEnum(enum.Enum):
    """CallableEnum class"""

    def __call__(self, x):
        return self.value(x)
