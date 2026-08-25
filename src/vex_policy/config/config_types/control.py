"""Policy control component configuration."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ControlModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InputParameter(ControlModel):
    name: str
    min: float
    max: float
    default: float

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        if not value or value.strip() != value:
            raise ValueError("input parameter name must be non-empty and have no surrounding whitespace")
        return value

    @model_validator(mode="after")
    def validate_range(self) -> InputParameter:
        if not all(math.isfinite(value) for value in (self.min, self.max, self.default)):
            raise ValueError("input parameter bounds and default must be finite")
        if self.min >= self.max:
            raise ValueError("input parameter min must be less than max")
        if not self.min <= self.default <= self.max:
            raise ValueError("input parameter default must be inside [min, max]")
        return self


class JoystickInput(ControlModel):
    type: Literal["joystick"]
    x: InputParameter
    y: InputParameter

    @model_validator(mode="after")
    def distinct_axes(self) -> JoystickInput:
        if self.x.name == self.y.name:
            raise ValueError("joystick axes must control different parameters")
        return self


class SliderInput(ControlModel):
    type: Literal["slider"]
    parameter: InputParameter


PolicyInput = Annotated[JoystickInput | SliderInput, Field(discriminator="type")]


def input_parameters(inputs: tuple[PolicyInput, ...]) -> tuple[InputParameter, ...]:
    """Flatten ordered UI components into their ordered parameters."""
    parameters: list[InputParameter] = []
    for component in inputs:
        if isinstance(component, JoystickInput):
            parameters.extend((component.x, component.y))
        else:
            parameters.append(component.parameter)
    return tuple(parameters)
