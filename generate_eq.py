import numpy as np
import pandas as pd

DAMAGE_STATES = ['none', 'minor', 'severe', 'collapse']

def generate_buildings(n_buildings, diameter_km, seed=42):

    np.random.seed(seed)
    buildings = []
    
    radius = diameter_km / 2
    
    for i in range(n_buildings):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0.5, radius)
        
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        distance = r
        
        building_type = np.random.choice(
            ['wood', 'concrete', 'steel'], 
            p=[0.5, 0.3, 0.2]
        )
        
        if building_type == 'wood':
            occupancy = int(np.random.uniform(10, 100))
        elif building_type == 'concrete':
            occupancy = int(np.random.uniform(50, 300))
        else:
            occupancy = int(np.random.uniform(100, 500))
        
        buildings.append({
            'building_id': i,
            'x': round(x, 2),
            'y': round(y, 2),
            'distance_km': round(distance, 2),
            'building_type': building_type,
            'occupancy': occupancy
        })
    
    return pd.DataFrame(buildings)

def compute_damage_probabilities(distance, building_type, magnitude, alpha_params):
    magnitude_scale = (magnitude - 5.0) / 3.0
    magnitude_scale = max(0.1, min(magnitude_scale, 2.0))
    
    base_damage = np.exp(-distance / (15.0 * magnitude_scale))
    
    alpha = alpha_params[building_type]
    
    p_collapse = alpha * base_damage * 0.30 * magnitude_scale
    p_severe = alpha * base_damage * 0.25 * magnitude_scale
    p_minor = alpha * base_damage * 0.20 * magnitude_scale
    
    total = p_collapse + p_severe + p_minor
    if total > 0.95:
        scale = 0.95 / total
        p_collapse *= scale
        p_severe *= scale
        p_minor *= scale
    
    p_none = 1 - (p_collapse + p_severe + p_minor)
    
    return np.array([p_none, p_minor, p_severe, p_collapse])

def simulate_damage(buildings_df, magnitude, alpha_params, seed=42):
    np.random.seed(seed)
    
    buildings = buildings_df.copy()
    
    for idx, building in buildings.iterrows():
        probs = compute_damage_probabilities(
            building['distance_km'],
            building['building_type'],
            magnitude,
            alpha_params
        )
        
        damage_state = np.random.choice(DAMAGE_STATES, p=probs)
        
        buildings.at[idx, 'true_damage'] = damage_state
        buildings.at[idx, 'p_none'] = round(probs[0], 4)
        buildings.at[idx, 'p_minor'] = round(probs[1], 4)
        buildings.at[idx, 'p_severe'] = round(probs[2], 4)
        buildings.at[idx, 'p_collapse'] = round(probs[3], 4)
    
    return buildings

def create_scenario(n_buildings=100, diameter_km=40, magnitude=6.5, 
                   alpha_params=None, seed=42):
    if alpha_params is None:
        alpha_params = {'wood': 1.5, 'concrete': 1.0, 'steel': 0.7}
    
    buildings = generate_buildings(n_buildings, diameter_km, seed)
    scenario = simulate_damage(buildings, magnitude, alpha_params, seed)
    
    return scenario