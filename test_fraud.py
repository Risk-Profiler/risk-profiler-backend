from api.schemas import RiskInput
from api.ml_service import predict_risk
import json

def run_test(name, data: RiskInput):
    print(f"\n{'='*60}")
    print(f"Testing Scenario: {name}")
    print(f"{'='*60}")
    
    result = predict_risk(data)
    
    print(f"Risk Level: {result['risk_level']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Recommended Limit: Rp {result['recommended_limit']:,.0f}")
    
    print("\n--- Fraud Analysis ---")
    print(json.dumps(result['fraud_analysis'], indent=2))
    
    print("\n--- Conventional Recommendations ---")
    for rec in result['conventional_recommendations']:
        print(f" - {rec}")


if __name__ == "__main__":
    # Test 1: Valid Merchant
    valid_data = RiskInput(
        merchant_id='MID-VALID',
        business_age_months=24,
        qris_volume_monthly=20000000,
        qris_active_days=25,
        ecommerce_rating=4.5,
        pln_delay_days=0,
        pdam_bill_avg=150000,
        pdam_late_payments=0,
        business_category='retail',
        ktp_name="Joko Anwar",
        qris_name="Toko Joko Anwar",
        ecommerce_name="Joko Retail",
        application_lat=-6.200000,
        application_lon=106.816666, # Jakarta
        store_lat=-6.205000,
        store_lon=106.820000, # Jakarta (very close)
        recent_weekly_volume=5000000, # 25% of monthly, normal
        out_of_hours_ratio=0.05, # Normal
        round_amount_ratio=0.10 # Normal
    )
    
    # Test 2: Identity Theft & Location Anomaly
    id_theft_data = RiskInput(
        merchant_id='MID-IDTHEFT',
        business_age_months=24,
        qris_volume_monthly=20000000,
        qris_active_days=25,
        ecommerce_rating=4.5,
        pln_delay_days=0,
        pdam_bill_avg=150000,
        pdam_late_payments=0,
        business_category='retail',
        ktp_name="Budi Santoso",
        qris_name="Warung Ayu",
        ecommerce_name="Ayu Store",
        application_lat=1.3521,
        application_lon=103.8198, # Singapore
        store_lat=-6.200000,
        store_lon=106.816666, # Jakarta
    )
    
    # Test 3: Gestun & AML Velocity
    gestun_data = RiskInput(
        merchant_id='MID-GESTUN',
        business_age_months=24,
        qris_volume_monthly=15000000,
        qris_active_days=25,
        ecommerce_rating=4.5,
        pln_delay_days=0,
        pdam_bill_avg=150000,
        pdam_late_payments=0,
        business_category='retail',
        ktp_name="Andi",
        qris_name="Andi Cell",
        recent_weekly_volume=30000000, # Spike! 2x the entire monthly volume
        out_of_hours_ratio=0.45, # 45% out of hours!
        round_amount_ratio=0.85 # 85% round amounts! Gestun!
    )
    
    run_test("Valid Merchant (Control)", valid_data)
    run_test("Identity Theft & Location Anomaly", id_theft_data)
    run_test("Gestun & Velocity Spike", gestun_data)
    
    print("\nAll anti-fraud tests completed successfully!")
