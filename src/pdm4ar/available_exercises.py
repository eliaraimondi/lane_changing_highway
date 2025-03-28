from pdm4ar.exercises_def import *
from pdm4ar.exercises_def.structures import Exercise

available_exercises: Mapping[str, Callable[[], Exercise]] = frozendict(
    {
        "01": get_exercise1,
        "02": get_exercise2,
        "03": get_exercise3,
        "04": get_exercise4,
        "05": get_exercise5,
        "06": get_exercise6,
        "07": get_exercise7,
        "08": get_exercise8,
        "09": get_exercise09,
        "10": get_exercise10,
        "11": get_exercise11,
        "12": get_exercise12
    }
)
