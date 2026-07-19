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
    
    # Anti-Fraud & AML Fields (Optional)
    ktp_name: str | None = None
    qris_name: str | None = None
    ecommerce_name: str | None = None
    
    application_lat: float | None = None
    application_lon: float | None = None
    store_lat: float | None = None
    store_lon: float | None = None
    
    recent_weekly_volume: float | None = None
    out_of_hours_ratio: float | None = None
    round_amount_ratio: float | None = None


class DecisionInput(BaseModel):
    merchant_id: str
    status: Literal["Approved", "Rejected", "Revision Requested"]
    note: str | None = None
    revision_limit: float | None = None
