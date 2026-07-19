def add_featured_engineering(df):
    df = df.copy()
    
    df['qris_active_ratio'] = df['qris_active_days'] / 30
    df['pln_delay_ratio'] = df['pln_delay_days'] / (df['business_age_months'] + 1)
    df['volume_to_age_ratio'] = df['qris_volume_monthly'] / (df['business_age_months'] + 1)
    df['chronic_pln_delay'] = (df['pln_delay_days'] > 14).astype(int)
    df['has_both_utilities'] = (1 - df['pln_delay_days_isna']) * (1 - df['pdam_bill_avg_isna'])

    return df