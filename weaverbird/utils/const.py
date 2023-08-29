from enum import Enum


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    def __str__(self):
        return str(self.value)

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )

class LogConst(ExplicitEnum):
    """Format for log handler.
    """
    DEFAULT_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s'
    DEFAULT_FORMAT_LONG = '%(asctime)s - %(filename)s[pid:%(process)d;line:%(lineno)d:%(funcName)s]' \
                          ' - %(levelname)s: %(message)s'


class Language(ExplicitEnum):
    EN = 'en'
    CN = 'cn'