"""
Shared description of a selectable technique.

Every pipeline stage (chunking, embedding, retrieval) exposes a registry of
`Technique`s. The UI renders a picker from the registry and builds parameter
widgets from each technique's `params`, so adding a new technique is just a
matter of adding an entry -- no UI changes needed.
"""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Param:
    name: str
    label: str
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None  # set -> rendered as a dropdown
    help: str = ""


@dataclass
class Technique:
    name: str
    description: str
    build: Callable[..., Any]
    params: list[Param] = field(default_factory=list)

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}
