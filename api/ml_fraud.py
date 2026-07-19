import math
import re
from api.schemas import RiskInput

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in kilometers between two points on the earth."""
    R = 6371.0 # Earth radius in kilometers
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def tokenize_name(name: str):
    """Lowercase and extract alphanumeric words from a name."""
    if not name:
        return set()
    words = re.findall(r'\b\w+\b', name.lower())
    return set(words)

def check_identity_mismatch(data: RiskInput):
    if not data.ktp_name:
        return None
    
    ktp_tokens = tokenize_name(data.ktp_name)
    flags = []
    
    if data.qris_name:
        qris_tokens = tokenize_name(data.qris_name)
        if len(ktp_tokens.intersection(qris_tokens)) == 0:
            flags.append(f"Identity Risk: KTP Name '{data.ktp_name}' does not match QRIS Name '{data.qris_name}'")
            
    if data.ecommerce_name:
        ecom_tokens = tokenize_name(data.ecommerce_name)
        if len(ktp_tokens.intersection(ecom_tokens)) == 0:
            flags.append(f"Identity Risk: KTP Name '{data.ktp_name}' does not match E-Commerce Name '{data.ecommerce_name}'")
            
    return flags if flags else None

def check_location_anomaly(data: RiskInput):
    if None in (data.application_lat, data.application_lon, data.store_lat, data.store_lon):
        return None
        
    distance = haversine_distance(data.application_lat, data.application_lon, data.store_lat, data.store_lon)
    if distance > 50.0:
        return f"Suspicious Location: Application submitted {distance:.1f} km away from registered store location."
    return None

def check_aml_velocity(data: RiskInput):
    if data.recent_weekly_volume is None or data.qris_volume_monthly is None or data.qris_volume_monthly == 0:
        return None
        
    # If weekly volume is > 50% of the entire monthly volume expectation, flag it as a spike
    expected_weekly = data.qris_volume_monthly / 4.0
    if data.recent_weekly_volume > expected_weekly * 2.0:
        return f"Velocity Anomaly: Recent weekly volume (Rp {data.recent_weekly_volume:,.0f}) is unusually high compared to monthly average (Rp {data.qris_volume_monthly:,.0f})."
    return None

def check_out_of_hours(data: RiskInput):
    if data.out_of_hours_ratio is None:
        return None
    
    if data.out_of_hours_ratio > 0.30:
        return f"Out-of-Hours Anomaly: {data.out_of_hours_ratio:.0%} of transactions occurred between 1 AM and 4 AM."
    return None

def check_gestun_fraud(data: RiskInput):
    if data.round_amount_ratio is None:
        return None
        
    if data.round_amount_ratio > 0.40:
        return f"Gestun Indicator: Unusually high percentage ({data.round_amount_ratio:.0%}) of transactions are exact round amounts."
    return None

def evaluate_fraud_rules(data: RiskInput):
    flags = []
    
    identity_flags = check_identity_mismatch(data)
    if identity_flags:
        flags.extend(identity_flags)
        
    loc_flag = check_location_anomaly(data)
    if loc_flag:
        flags.append(loc_flag)
        
    vel_flag = check_aml_velocity(data)
    if vel_flag:
        flags.append(vel_flag)
        
    ooh_flag = check_out_of_hours(data)
    if ooh_flag:
        flags.append(ooh_flag)
        
    gestun_flag = check_gestun_fraud(data)
    if gestun_flag:
        flags.append(gestun_flag)
        
    return {
        "is_suspicious": len(flags) > 0,
        "flags": flags
    }
