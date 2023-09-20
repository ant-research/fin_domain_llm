from dataclasses import dataclass, asdict


@dataclass
class BaseConfig:

    def dict(self):
        return {k: v for k, v in asdict(self).items()}
