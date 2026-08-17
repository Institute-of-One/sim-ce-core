"""Physiology and injection-protocol parameters (Bae-style reduced model)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InjectionProtocol(BaseModel):
    """Rectangular iodinated-contrast bolus."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    concentration_mgi_ml: float = Field(
        gt=0, description="Iodine concentration (mg I / mL)."
    )
    volume_ml: float = Field(gt=0, description="Injected volume (mL).")
    duration_s: float = Field(gt=0, description="Injection duration (s).")

    @property
    def rate_ml_s(self) -> float:
        return self.volume_ml / self.duration_s

    @property
    def iodine_rate_mgi_s(self) -> float:
        return self.rate_ml_s * self.concentration_mgi_ml

    @property
    def iodine_mass_mgi(self) -> float:
        return self.volume_ml * self.concentration_mgi_ml


class PhysioParams(BaseModel):
    """Reduced Bae-style physiology: central blood, organ, recirculation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    central_blood_volume_ml: float = Field(
        gt=0, description="Central / mixing blood volume (mL)."
    )
    organ_volume_ml: float = Field(gt=0, description="Organ distribution volume (mL).")
    recirculation_volume_ml: float = Field(
        gt=0, description="Recirculation / remainder volume (mL)."
    )
    cardiac_output_ml_s: float = Field(gt=0, description="Cardiac output (mL/s).")
    organ_flow_fraction: float = Field(
        gt=0,
        lt=1,
        description="Fraction of cardiac output to the organ compartment.",
    )
    elimination_rate_1_s: float = Field(
        default=0.0,
        ge=0,
        description="First-order iodine elimination from recirculation (1/s).",
    )
    iodine_to_hu: float = Field(
        default=26.0,
        gt=0,
        description="Attenuation conversion (HU per mg I / mL).",
    )
    transit_delay_s: float = Field(
        default=0.0,
        ge=0,
        description="Bolus delay from injection site to central blood (s).",
    )

    @property
    def organ_flow_ml_s(self) -> float:
        return self.cardiac_output_ml_s * self.organ_flow_fraction

    @property
    def recirculation_flow_ml_s(self) -> float:
        return self.cardiac_output_ml_s * (1.0 - self.organ_flow_fraction)


REGION_NAMES: tuple[str, str, str] = ("aorta", "organ", "recirculation")
