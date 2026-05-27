"""Domain grouping and authoritative source mapping for requested predictors.

This file is the canonical "Step 2: Select Predictor Variables" catalog.
It documents which variables are currently fetched automatically (via API),
and which require keys or local systems (DHIS2/CRVS/HMIS).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariableSpec:
    name: str
    domain: str
    unit: str | None
    primary_source: str
    api_hint: str | None = None
    notes: str | None = None


VARIABLES: list[VariableSpec] = [
    # Health system variables
    VariableSpec(
        name="Skilled birth attendance (%)",
        domain="Health system",
        unit="percent",
        primary_source="World Bank / WHO (via World Bank API)",
        api_hint="World Bank indicator SH.STA.BRTW.ZS",
    ),
    VariableSpec(
        name="Caesarean section rate",
        domain="Health system",
        unit="percent",
        primary_source="WHO / UNICEF (varies by dataset)",
        notes="Not consistently available via one open global API; often DHS/MICS/WHO.",
    ),
    VariableSpec(
        name="Facility delivery rate",
        domain="Health system",
        unit="percent",
        primary_source="WHO / UNICEF (often DHS/MICS)",
    ),
    VariableSpec(
        name="Health workforce density",
        domain="Health system",
        unit="per 10,000",
        primary_source="WHO GHO",
        notes="Requires WHO GHO indicator selection.",
    ),
    VariableSpec(
        name="Midwife density",
        domain="Health system",
        unit="per 10,000",
        primary_source="WHO GHO",
        notes="Requires WHO GHO indicator selection.",
    ),
    VariableSpec(
        name="Emergency obstetric care coverage",
        domain="Health system",
        unit="percent",
        primary_source="WHO / UNFPA (varies)",
        notes="Often from facility assessments; not a single global open API.",
    ),
    VariableSpec(
        name="Blood availability",
        domain="Health system",
        unit=None,
        primary_source="Country logistics / blood services",
        notes="Typically not available globally; often national systems.",
    ),
    VariableSpec(
        name="Referral efficiency",
        domain="Health system",
        unit=None,
        primary_source="Country health system (HMIS/DHIS2)",
        notes="Requires country HMIS metrics definition.",
    ),

    # Reproductive variables
    VariableSpec(
        name="Contraceptive prevalence",
        domain="Reproductive",
        unit="percent",
        primary_source="World Bank / UNICEF / UNFPA (varies by indicator)",
        notes="Often available in World Bank as contraceptive prevalence (married women).",
    ),
    VariableSpec(
        name="Adolescent fertility",
        domain="Reproductive",
        unit="births per 1,000 women 15-19",
        primary_source="World Bank / UNFPA",
        api_hint="World Bank indicator SP.ADO.TFRT",
    ),
    VariableSpec(
        name="Birth spacing",
        domain="Reproductive",
        unit=None,
        primary_source="DHS / MICS",
        notes="Usually computed from microdata; not global API.",
    ),
    VariableSpec(
        name="ANC4/ANC8 coverage",
        domain="Reproductive",
        unit="percent",
        primary_source="WHO / UNICEF (often DHS/MICS + modeled series)",
        notes="GHO / UNICEF SDG APIs may provide ANC4; ANC8 is less consistently exposed.",
    ),

    # Disease variables
    VariableSpec(
        name="HIV prevalence",
        domain="Disease",
        unit="percent",
        primary_source="UNAIDS / WHO",
        notes="Often via UNAIDS; WHO GHO has related series in some cases.",
    ),
    VariableSpec(
        name="Malaria prevalence",
        domain="Disease",
        unit=None,
        primary_source="WHO / Malaria Atlas Project (varies)",
    ),
    VariableSpec(
        name="Anaemia prevalence",
        domain="Disease",
        unit="percent",
        primary_source="WHO",
    ),

    # Socioeconomic variables
    VariableSpec(
        name="Female literacy",
        domain="Socioeconomic",
        unit="percent",
        primary_source="World Bank / UNESCO (via World Bank API)",
        api_hint="World Bank indicator SE.ADT.LITR.FE.ZS",
    ),
    VariableSpec(
        name="General population literacy rates",
        domain="Socioeconomic",
        unit="percent",
        primary_source="World Bank / UNESCO",
        notes="Often SE.ADT.LITR.ZS.",
    ),
    VariableSpec(
        name="Poverty rate",
        domain="Socioeconomic",
        unit="percent",
        primary_source="World Bank (PovcalNet/World Bank indicators)",
        notes="Needs poverty definition (e.g., $2.15/day or national poverty line).",
    ),
    VariableSpec(
        name="Rural population",
        domain="Socioeconomic",
        unit="percent",
        primary_source="World Bank",
        api_hint="World Bank indicator SP.RUR.TOTL.ZS",
    ),
    VariableSpec(
        name="Gender inequality index",
        domain="Socioeconomic",
        unit=None,
        primary_source="UNDP (not UNICEF/UNSD API by default)",
        notes="Often distributed by UNDP; may require separate download.",
    ),
    VariableSpec(
        name="Fertility rate",
        domain="Socioeconomic",
        unit="births per woman",
        primary_source="World Bank / UNFPA",
        api_hint="World Bank indicator SP.DYN.TFRT.IN",
    ),
    VariableSpec(
        name="Adolescent fertility rates",
        domain="Socioeconomic",
        unit="births per 1,000 women 15-19",
        primary_source="World Bank / UNFPA",
        api_hint="World Bank indicator SP.ADO.TFRT",
    ),

    # Infrastructure variables
    VariableSpec(
        name="Travel time to facility",
        domain="Infrastructure",
        unit="minutes",
        primary_source="AccessMod / national GIS",
        notes="Not typically available as a global API; requires spatial modeling.",
    ),
    VariableSpec(
        name="Road access",
        domain="Infrastructure",
        unit=None,
        primary_source="WorldPop/OSM-derived metrics (varies)",
    ),
    VariableSpec(
        name="Ambulance coverage",
        domain="Infrastructure",
        unit=None,
        primary_source="Country systems / facility assessments",
    ),

    # Governance / data variables
    VariableSpec(
        name="Total government health expenditure",
        domain="Governance & data",
        unit="percent of GDP or per capita",
        primary_source="World Bank / WHO GHED",
        notes="Needs indicator choice (e.g., GGHE-D as % of GDP).",
    ),
    VariableSpec(
        name="Civil registration completeness",
        domain="Governance & data",
        unit="percent",
        primary_source="UNSD / CRVS / World Bank (varies)",
        notes="Often not a single clean global API; may require UNSD/CRVS datasets.",
    ),
    VariableSpec(
        name="Health information system maturity",
        domain="Governance & data",
        unit=None,
        primary_source="Country HIS assessments",
    ),

    # Emergency & shocks
    VariableSpec(
        name="Epidemic occurrence",
        domain="Emergency & shocks",
        unit=None,
        primary_source="WHO outbreaks",
        notes="WHO provides feeds; indicator construction required (binary/count).",
    ),
    VariableSpec(
        name="Conflict intensity",
        domain="Emergency & shocks",
        unit=None,
        primary_source="ACLED",
        notes="Requires ACLED API key/terms; we can wire it in via env var.",
    ),
    VariableSpec(
        name="Flood occurrence",
        domain="Emergency & shocks",
        unit=None,
        primary_source="EM-DAT",
        notes="Requires EM-DAT access credentials; can be ingested if provided.",
    ),
    VariableSpec(
        name="Supply chain disruption",
        domain="Emergency & shocks",
        unit=None,
        primary_source="Country logistics systems",
    ),
    VariableSpec(
        name="Workforce strike duration",
        domain="Emergency & shocks",
        unit="days",
        primary_source="Government reports",
    ),
]


def domains() -> list[str]:
    return sorted({v.domain for v in VARIABLES})


def variables_by_domain() -> dict[str, list[VariableSpec]]:
    out: dict[str, list[VariableSpec]] = {d: [] for d in domains()}
    for v in VARIABLES:
        out[v.domain].append(v)
    return out

