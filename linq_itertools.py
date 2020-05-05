from typing import *

T = TypeVar('T')
def single_or_none(predicate, sequence: Sequence[T]) -> Optional[T]:
    filtered = [item for item in sequence if predicate(item)]
    if len(filtered) == 1:
        return filtered[0]
    else:
        return None