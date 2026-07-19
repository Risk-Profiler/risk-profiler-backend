from typing import Literal

from pydantic import BaseModel

class RiskInput(BaseModel):
    merchant_id: str
    business_age_months: int
    qris_volume_monthly: float
    qris_active_days: int
    ecommerce_rating: float | None = None
    pln_delay_days: int | None = None
    pdam_bill_avg: float | None = None
    pdam_late_payments: int | None = None
    business_category: str


class DecisionInput(BaseModel):
    merchant_id: str
    status: Literal["Approved", "Rejected", "Revision Requested"]
    note: str | None = None
    revision_limit: float | None = None
