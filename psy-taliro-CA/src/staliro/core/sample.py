from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence, Sized, Tuple, Union, cast

from attr import field, frozen
from numpy import float64, ndarray
from numpy.typing import NDArray
from typing_extensions import overload


def _value_converter(value: Any) -> float:
    if isinstance(value, int):
        return float(value)

    if isinstance(value, float):
        return value

    raise TypeError("only int and float are valid sample value types")


_ValuesT = Union[NDArray[Any], Sequence[int], Sequence[float]]


def _values_converter(value: _ValuesT) -> Tuple[float, ...]:
    if isinstance(value, ndarray):
        return cast(Tuple[float, ...], tuple(value.astype(float64).tolist()))

    if isinstance(value, (list, tuple)):
        return tuple(_value_converter(v) for v in value)

    raise TypeError("only ndarray, list and tuple are valid collections to create a sample")


@frozen(slots=True)
class Sample(Sized, Iterable[float]):
    """Representation of a sample generated by the optimizer.

    Args:
        values: The sample values

    Attributes:
        values: The sample values
    """

    values: Tuple[float, ...] = field(converter=_values_converter)

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[float]:
        return iter(self.values)

    @overload
    def __getitem__(self, index: int) -> float:
        ...

    @overload
    def __getitem__(self, index: slice) -> Tuple[float, ...]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[float, Tuple[float, ...]]:
        if isinstance(index, (int, slice)):
            return self.values[index]
        else:
            raise TypeError()
