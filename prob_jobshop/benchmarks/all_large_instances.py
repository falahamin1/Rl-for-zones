from typing import List

from ..instance import ProbJobShopInstance
from . import ljs_01, ljs_02, ljs_03, ljs_04, ljs_05
from . import ljs_06, ljs_07, ljs_08, ljs_09, ljs_10

ALL_LARGE_INSTANCE_NAMES = [
    "LJS_01", "LJS_02", "LJS_03", "LJS_04", "LJS_05",
    "LJS_06", "LJS_07", "LJS_08", "LJS_09", "LJS_10",
]

_GETTERS = [
    ljs_01.get_instance,
    ljs_02.get_instance,
    ljs_03.get_instance,
    ljs_04.get_instance,
    ljs_05.get_instance,
    ljs_06.get_instance,
    ljs_07.get_instance,
    ljs_08.get_instance,
    ljs_09.get_instance,
    ljs_10.get_instance,
]


def get_all_large_instances() -> List[ProbJobShopInstance]:
    return [g() for g in _GETTERS]


def get_large_instance_by_name(name: str) -> ProbJobShopInstance:
    idx = ALL_LARGE_INSTANCE_NAMES.index(name)
    return _GETTERS[idx]()
