from typing import List

from ..instance import ProbJobShopInstance
from . import pjs_01, pjs_02, pjs_03, pjs_04, pjs_05
from . import pjs_06, pjs_07, pjs_08, pjs_09, pjs_10

ALL_INSTANCE_NAMES = [
    "PJS_01", "PJS_02", "PJS_03", "PJS_04", "PJS_05",
    "PJS_06", "PJS_07", "PJS_08", "PJS_09", "PJS_10",
]

_GETTERS = [
    pjs_01.get_instance,
    pjs_02.get_instance,
    pjs_03.get_instance,
    pjs_04.get_instance,
    pjs_05.get_instance,
    pjs_06.get_instance,
    pjs_07.get_instance,
    pjs_08.get_instance,
    pjs_09.get_instance,
    pjs_10.get_instance,
]


def get_all_instances() -> List[ProbJobShopInstance]:
    return [g() for g in _GETTERS]


def get_instance_by_name(name: str) -> ProbJobShopInstance:
    idx = ALL_INSTANCE_NAMES.index(name)
    return _GETTERS[idx]()
