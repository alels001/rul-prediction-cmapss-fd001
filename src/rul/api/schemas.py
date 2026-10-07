"""Request and response models.

Pydantic checks every request against these definitions before any of our
code runs. A request with a missing sensor, a text value, an empty history
or too many cycles is rejected with HTTP 422 and a message naming the field.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rul import config


class SensorReading(BaseModel):
    """Raw readings of the 13 retained sensors at one operating cycle.

    Other columns of a C-MAPSS row (settings, the 8 unused sensors) may be
    sent as well; they are ignored. Infinite and NaN values are rejected.
    """

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    cycle: int | None = Field(default=None, ge=1, description="Cycle index (optional)")
    s2: float
    s3: float
    s4: float
    s7: float
    s8: float
    s9: float
    s11: float
    s12: float
    s13: float
    s15: float
    s17: float
    s20: float
    s21: float

    def values(self) -> list[float]:
        """The 13 readings in the fixed order the model expects."""
        return [getattr(self, s) for s in config.SENSORS]


class EngineHistory(BaseModel):
    """Every recorded cycle of one engine, oldest first."""

    engine_id: str | None = Field(default=None, max_length=64, examples=["engine-1"])
    cycles: list[SensorReading] = Field(min_length=1, max_length=config.MAX_HISTORY)

    @model_validator(mode="after")
    def cycles_in_order(self) -> EngineHistory:
        numbers = [c.cycle for c in self.cycles if c.cycle is not None]
        if numbers and len(numbers) != len(self.cycles):
            raise ValueError("give `cycle` for every reading or for none")
        if numbers and any(b <= a for a, b in zip(numbers, numbers[1:], strict=False)):
            raise ValueError("readings must be ordered by strictly increasing `cycle`")
        return self


class BatchRequest(BaseModel):
    engines: list[EngineHistory] = Field(min_length=1, max_length=100)


class Prediction(BaseModel):
    engine_id: str | None
    rul: float = Field(description="Predicted remaining useful life, in cycles")
    capped: bool = Field(
        description=(
            "True when the prediction sits at the cap: the model reads the engine as "
            "healthy, with at least this many cycles left."
        )
    )
    n_cycles_observed: int


class PredictionResponse(Prediction):
    model_version: str


class BatchResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    trained_at: str | None = None
    r_max: int
    sensors: list[str]
    training_data: dict | None = None
    test_metrics: dict | None = None
