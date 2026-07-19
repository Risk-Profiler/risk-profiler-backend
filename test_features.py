from api.schemas import RiskInput
from api.ml_service import predict_risk
import json

def test_merchant(name, data: RiskInput):
    print(f"\n{'='*50}")
    print(f"Testing Merchant: {name}")
    print(f"{'='*50}")
    result = predict_risk(data)
    
    print(f"Risk Level: {result['risk_level']}")
    print(f"Band: {result['band']}")
    print(f"Score: {result['score']}")
    print(f"PD (High Risk Prob): {result['class_probabilities'].get('High Risk (2)', 'N/A')}")
    print(f"Recommended Limit: Rp {result['recommended_limit']:,.0f}")
    
    print("\n--- Split Recommendations (Shariah) ---")
    for rec in result['shariah_recommendations']:
        print(f" - {rec}")
        
    print("\n--- Shariah Metrics ---")
    print(json.dumps(result['shariah_metrics'], indent=2))
    print("\n")


if __name__ == "__main__":
    # Test 1: Low Risk Merchant (Excellent Data)
    low_risk_data = RiskInput(
        merchant_id='MID-LOW',
        business_age_months=36,
        qris_volume_monthly=25000000,
        qris_active_days=28,
        ecommerce_rating=4.9,
        pln_delay_days=0,
        pdam_bill_avg=300000,
        pdam_late_payments=0,
        business_category='fnb'
    )
    
    # Test 2: Medium Risk Merchant
    med_risk_data = RiskInput(
        merchant_id='MID-MED',
        business_age_months=14,
        qris_volume_monthly=8000000,
        qris_active_days=16,
        ecommerce_rating=3.8,
        pln_delay_days=7,
        pdam_bill_avg=180000,
        pdam_late_payments=1,
        business_category='retail'
    )
    
    # Test 3: High Risk Merchant (Band E expectation)
    high_risk_data = RiskInput(
        merchant_id='MID-HIGH',
        business_age_months=2,
        qris_volume_monthly=1000000,
        qris_active_days=5,
        ecommerce_rating=2.1,
        pln_delay_days=30,
        pdam_bill_avg=50000,
        pdam_late_payments=5,
        business_category='fashion'
    )
    
    test_merchant("Low Risk Merchant", low_risk_data)
    test_merchant("Medium Risk Merchant", med_risk_data)
    test_merchant("High Risk Merchant", high_risk_data)
    
    print("All tests completed successfully!")
