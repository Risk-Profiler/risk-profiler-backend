from pydantic import BaseModel

class RiskInput(BaseModel):
    merchant_id: str
    business_age_months: int
    qris_volume_monthly: float
    qris_active_days: int
    ecommerce_rating: float
    pln_delay_days: int
    pdam_bill_avg: float
    pdam_late_payments: int
    business_category: str