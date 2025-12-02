import numpy as np
import pandas as pd

DAMAGE_STATES = ['none', 'minor', 'severe', 'collapse']

REPORT_SOURCES = {
    'automated_sensor': {'reliability': 0.85, 'weight': 0.25},
    'phone_call': {'reliability': 0.60, 'weight': 0.50},
    'social_media': {'reliability': 0.40, 'weight': 0.20},
    'inspector': {'reliability': 0.95, 'weight': 0.05}
}

def generate_noisy_report(true_state, reliability):
    if np.random.random() < reliability:
        return true_state
    
    other_states = [s for s in DAMAGE_STATES if s != true_state]
    true_idx = DAMAGE_STATES.index(true_state)
    
    weights = []
    for state in other_states:
        state_idx = DAMAGE_STATES.index(state)
        distance = abs(state_idx - true_idx)
        weight = 1.0 / (distance + 1)
        weights.append(weight)
    
    weights = np.array(weights) / sum(weights)
    return np.random.choice(other_states, p=weights)

def generate_building_reports(building, lambda_rates, max_time_hours=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    true_damage = building['true_damage']
    lambda_rate = lambda_rates[true_damage]
    
    reports = []
    current_time = 0
    max_time_minutes = max_time_hours * 60
    
    while current_time < max_time_minutes:
        lambda_per_minute = lambda_rate / 60.0
        time_to_next = np.random.exponential(1.0 / lambda_per_minute)
        current_time += time_to_next
        
        if current_time >= max_time_minutes:
            break
        
        source_types = list(REPORT_SOURCES.keys())
        source_weights = [REPORT_SOURCES[s]['weight'] for s in source_types]
        source = np.random.choice(source_types, p=source_weights)
        
        reliability = REPORT_SOURCES[source]['reliability']
        reported_state = generate_noisy_report(true_damage, reliability)
        
        reports.append({
            'time_minutes': round(current_time, 2),
            'source': source,
            'reported_state': reported_state,
            'building_id': building['building_id'],
            'building_type': building['building_type'],
            'true_damage': building['true_damage']
        })
    
    return reports

def generate_all_reports(buildings_df, lambda_rates=None, max_time_hours=3, seed=42):
    if lambda_rates is None:
        lambda_rates = {
            'collapse': 8.0,
            'severe': 3.0,
            'minor': 0.8,
            'none': 0.2
        }
    
    np.random.seed(seed)
    all_reports = []
    
    for idx, building in buildings_df.iterrows():
        building_seed = seed + idx if seed is not None else None
        reports = generate_building_reports(
            building.to_dict(), 
            lambda_rates, 
            max_time_hours,
            building_seed
        )
        all_reports.extend(reports)
    
    reports_df = pd.DataFrame(all_reports)
    if len(reports_df) > 0:
        reports_df = reports_df.sort_values('time_minutes').reset_index(drop=True)
    
    return reports_df

def get_report_reliability(source):
    return REPORT_SOURCES[source]['reliability']