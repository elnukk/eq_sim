import numpy as np
import pandas as pd
from generate_reports import get_report_reliability

DAMAGE_STATES = ['none', 'minor', 'severe', 'collapse']

def compute_likelihood(reported_state, true_state, reliability):
    if reported_state == true_state:
        return reliability
    
    reported_idx = DAMAGE_STATES.index(reported_state)
    true_idx = DAMAGE_STATES.index(true_state)
    distance = abs(reported_idx - true_idx)
    
    base_error_prob = (1 - reliability) / 3.0
    
    if distance == 1:
        return base_error_prob * 2.0
    elif distance == 2:
        return base_error_prob * 1.0
    else:
        return base_error_prob * 0.5

def bayesian_update(prior, reported_state, reliability):
    likelihood = np.array([
        compute_likelihood(reported_state, state, reliability)
        for state in DAMAGE_STATES
    ])
    
    numerator = likelihood * prior
    denominator = np.sum(numerator)
    
    if denominator < 1e-10:
        return prior
    
    posterior = numerator / denominator
    return posterior

def entropy(probs):
    probs = np.array(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def bootstrap_beliefs(reports, prior, n_bootstrap=100):
    if len(reports) == 0:
        return {
            'mean': prior,
            'std_dev': np.zeros_like(prior)
        }
    
    bootstrap_posteriors = []
    
    for _ in range(n_bootstrap):
        resampled_reports = [reports[i] for i in np.random.choice(
            len(reports), size=len(reports), replace=True
        )]
        
        belief = prior.copy()
        for report in resampled_reports:
            reliability = get_report_reliability(report['source'])
            belief = bayesian_update(belief, report['reported_state'], reliability)
        
        bootstrap_posteriors.append(belief)
    
    bootstrap_posteriors = np.array(bootstrap_posteriors)
    
    return {
        'mean': np.mean(bootstrap_posteriors, axis=0),
        'std_dev': np.std(bootstrap_posteriors, axis=0) 
    }

def process_building(building, all_reports, prior):
    building_id = building['building_id']
    building_reports = all_reports[all_reports['building_id'] == building_id]
    building_reports = building_reports.sort_values('time_minutes')
    
    current_belief = prior.copy()
    report_list = []
    
    for _, report in building_reports.iterrows():
        reliability = get_report_reliability(report['source'])
        current_belief = bayesian_update(
            current_belief, 
            report['reported_state'], 
            reliability
        )
        report_list.append(report.to_dict())
    
    bootstrap_result = bootstrap_beliefs(report_list, prior, n_bootstrap=50)
    
    return {
        'building_id': building_id,
        'p_none': current_belief[0],
        'p_minor': current_belief[1],
        'p_severe': current_belief[2],
        'p_collapse': current_belief[3],
        'entropy': entropy(current_belief),
        'num_reports': len(building_reports)
    }

def run_inference(buildings_df, reports_df, n_samples=5000):
    results = []

    for _, building in buildings_df.iterrows():
        prior = np.array([
            building['p_none'],
            building['p_minor'],
            building['p_severe'],
            building['p_collapse']
        ])

        result = process_building(building, reports_df, prior)

        # Normalize posterior before sampling
        posterior = np.array([
            result['p_none'],
            result['p_minor'],
            result['p_severe'],
            result['p_collapse']
        ])
        posterior = posterior / posterior.sum()  

        samples = np.random.choice([0,1,2,3], size=n_samples, p=posterior)
        collapse_samples = (samples == 3).astype(float)
        result['p_collapse_std'] = collapse_samples.std()

        results.append(result)

    return pd.DataFrame(results)


def compute_decision_metrics(buildings_df, beliefs_df, n_teams):
    merged = buildings_df[['building_id', 'true_damage', 'occupancy', 'p_none', 'p_minor', 'p_severe', 'p_collapse']].merge(
        beliefs_df[['building_id', 'p_none', 'p_minor', 'p_severe', 'p_collapse']],
        on='building_id',
        suffixes=('_prior', '_posterior')
    )
    
    merged['at_risk_true'] = merged.apply(lambda row: {
        'collapse': 0.9 * row['occupancy'],
        'severe': 0.4 * row['occupancy'],
        'minor': 0.05 * row['occupancy'],
        'none': 0
    }[row['true_damage']], axis=1)
    
    merged['expected_at_risk'] = (
        merged['p_collapse_posterior'] * 0.9 * merged['occupancy'] +
        merged['p_severe_posterior'] * 0.4 * merged['occupancy'] +
        merged['p_minor_posterior'] * 0.05 * merged['occupancy']
    )
    
    bayesian_top = merged.nlargest(n_teams, 'expected_at_risk')
    bayesian_saved = bayesian_top['at_risk_true'].sum()
    
    naive_top = merged.nlargest(n_teams, 'p_collapse_prior')
    naive_saved = naive_top['at_risk_true'].sum()
    
    return {
        'bayesian_lives_saved': int(bayesian_saved),
        'naive_lives_saved': int(naive_saved),
        'improvement': int(bayesian_saved - naive_saved),
        'improvement_pct': (bayesian_saved - naive_saved) / naive_saved * 100 if naive_saved > 0 else 0
    }