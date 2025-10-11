import abc
import typing as T

import pydantic as pdt

Locals = dict[str, T.Any]


class Job(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    @abc.abstractmethod
    def run(self) -> Locals:
        """Run the job in context.

        Returns:
            Locals: local job variables.
        """
