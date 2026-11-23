"""Copy/serialization contract shared by immutable extension values."""
from typing import NoReturn, Self, SupportsIndex


class _ImmutableValue:
    __slots__ = ()

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        del memo
        return self

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("pickle is unsupported for immutable mathematical values")
