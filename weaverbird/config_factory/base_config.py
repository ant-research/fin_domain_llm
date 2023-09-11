from dataclasses import dataclass, asdict


@dataclass
class BaseConfig:

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
