import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from generate_eq import create_scenario
from generate_reports import generate_all_reports
from update_bayesian import run_inference, compute_decision_metrics

st.set_page_config(layout="wide", page_title="Earthquake Response Simulator", page_icon="🌍")

st.title("Bayesian Earthquake Response Simulator")
st.markdown("Design your own earthquake scenario and watch Bayesian inference optimize rescue decisions")

with st.sidebar:
    st.header("Scenario Configuration")
    
    st.subheader("Earthquake Parameters")
    magnitude = st.slider("Magnitude (Richter)", 5.0, 8.0, 6.5, 0.1,
                         help="Higher magnitude = more energy, wider damage radius")
    diameter = st.slider("Affected Area Diameter (km)", 10, 100, 40, 5)
    n_buildings = st.slider("Number of Buildings", 20, 2000, 100, 10,
                           help="More buildings = more realistic but slower computation")
    n_teams = st.slider("Rescue Teams Available", 1, 50, 5, 1)
    
    st.subheader("Building Vulnerability")
    with st.expander("Advanced: Vulnerability Parameters"):
        alpha_wood = st.slider("Wood Buildings (α)", 0.5, 2.5, 1.5, 0.1)
        alpha_concrete = st.slider("Concrete Buildings (α)", 0.5, 2.5, 1.0, 0.1)
        alpha_steel = st.slider("Steel Buildings (α)", 0.5, 2.5, 0.7, 0.1)
    
    st.subheader("Report Arrival Rates")
    with st.expander("Advanced: Poisson Rates (per hour)"):
        lambda_collapse = st.slider("Collapsed Buildings", 1.0, 15.0, 8.0, 0.5)
        lambda_severe = st.slider("Severe Damage", 0.5, 8.0, 3.0, 0.5)
        lambda_minor = st.slider("Minor Damage", 0.1, 3.0, 0.8, 0.1)
        lambda_none = st.slider("No Damage", 0.05, 1.0, 0.2, 0.05)
    
    st.divider()
    
    sim_seed = st.number_input("Random Seed", 0, 9999, 42, 1)
    
    if st.button("Generate New Scenario", type="primary"):
        st.session_state.clear()
        st.rerun()

alpha_params = {'wood': alpha_wood, 'concrete': alpha_concrete, 'steel': alpha_steel}
lambda_rates = {'collapse': lambda_collapse, 'severe': lambda_severe, 'minor': lambda_minor, 'none': lambda_none}

if 'scenario' not in st.session_state:
    with st.spinner("Generating earthquake scenario..."):
        st.session_state.scenario = create_scenario(
            n_buildings=n_buildings,
            diameter_km=diameter,
            magnitude=magnitude,
            alpha_params=alpha_params,
            seed=sim_seed
        )
        st.session_state.reports = generate_all_reports(
            st.session_state.scenario,
            lambda_rates=lambda_rates,
            max_time_hours=3,
            seed=sim_seed
        )
        st.session_state.beliefs = run_inference(
            st.session_state.scenario,
            st.session_state.reports
        )
        st.session_state.metrics = compute_decision_metrics(
            st.session_state.scenario,
            st.session_state.beliefs,
            n_teams
        )

scenario = st.session_state.scenario
reports = st.session_state.reports
beliefs = st.session_state.beliefs
metrics = st.session_state.metrics

tab1, tab2, tab3 = st.tabs(["Live Simulation", "Building Analysis", "Results & Impact"])

with tab1:

    # --- Create placeholders so we control layout order ---
    metrics_placeholder = st.empty()
    st.divider()
    slider_placeholder = st.empty()

    # --- SLIDER RENDERED BELOW METRICS ---
    with slider_placeholder:
        current_time = st.slider(
            "Simulation Time (minutes)",
            min_value=0.0,
            max_value=180.0,
            value=0.0,
            step=1.0,
            help="Slide to watch how beliefs update as reports arrive over time"
        )

    # --- Now compute everything based on current_time ---
    current_reports = reports[reports['time_minutes'] <= current_time]
    num_current_reports = len(current_reports)

    buildings_reported = current_reports['building_id'].nunique()

    if num_current_reports > 0:
        buildings_with_reports = current_reports['building_id'].unique()
        current_beliefs_subset = beliefs[beliefs['building_id'].isin(buildings_with_reports)]
        avg_entropy = current_beliefs_subset['entropy'].mean()
    else:
        avg_entropy = None

    # --- RENDER METRICS ABOVE SLIDER ---
    with metrics_placeholder.container():
        col2, col3, col4 = st.columns(3)

        with col2:
            st.metric("Reports Received", num_current_reports)

        with col3:
            st.metric("Buildings Reported", buildings_reported)

        with col4:
            if avg_entropy is not None:
                st.metric("Avg Uncertainty", f"{avg_entropy:.2f}",
                          help="0 = certain, 2 = maximum uncertainty")
            else:
                st.metric("Avg Uncertainty", "—")

    
    from update_bayesian import bayesian_update
    from generate_reports import get_report_reliability
    
    beliefs_at_time = []
    for _, building in scenario.iterrows():
        prior = np.array([
            building['p_none'],
            building['p_minor'],
            building['p_severe'],
            building['p_collapse']
        ])
        
        building_reports = current_reports[current_reports['building_id'] == building['building_id']]
        current_belief = prior.copy()
        
        for _, report in building_reports.iterrows():
            reliability = get_report_reliability(report['source'])
            current_belief = bayesian_update(current_belief, report['reported_state'], reliability)
        
        beliefs_at_time.append({
            'building_id': building['building_id'],
            'p_collapse_current': current_belief[3],
            'has_reports': len(building_reports) > 0
        })
    
    beliefs_current_df = pd.DataFrame(beliefs_at_time)
    
    map_data = scenario[['building_id', 'x', 'y', 'distance_km', 'building_type', 'occupancy', 'true_damage']].merge(
        beliefs_current_df,
        on='building_id'
    )
    
    fig_map = px.scatter(
        map_data,
        x='x',
        y='y',
        size='occupancy',
        color='p_collapse_current',
        color_continuous_scale='RdYlGn_r',
        range_color=[0, 1],
        hover_data={
            'building_id': True,
            'building_type': True,
            'distance_km': ':.1f',
            'occupancy': True,
            'p_collapse_current': ':.3f',
            'true_damage': True,
            'has_reports': True,
            'x': False,
            'y': False
        },
        labels={'p_collapse_current': 'P(Collapse)', 'has_reports': 'Received Reports'},
        title=f"Building Risk Assessment at t={int(current_time)} minutes"
    )
    
    fig_map.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers+text',
        marker=dict(size=20, color='red', symbol='star'),
        text=['Epicenter'],
        textposition='top center',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    for radius in [5, 10, 15, 20]:
        if radius < diameter / 2:
            theta = np.linspace(0, 2*np.pi, 100)
            x_circle = radius * np.cos(theta)
            y_circle = radius * np.sin(theta)
            fig_map.add_trace(go.Scatter(
                x=x_circle, y=y_circle,
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig_map.update_layout(
        height=600,
        xaxis_title="Distance East-West (km)",
        yaxis_title="Distance North-South (km)",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    if num_current_reports > 0:
        with st.expander("Recent Reports", expanded=False):
            recent = current_reports.nsmallest(10, 'time_minutes', keep='last')
            recent = recent.sort_values('time_minutes', ascending=False)
            
            for _, report in recent.iterrows():
                st.text(
                    f"t={report['time_minutes']:6.1f}m | "
                    f"Building {report['building_id']:3.0f} | "
                    f"{report['source']:20s} | "
                    f"Reports: {report['reported_state']:8s} | "
                    f"True: {report['true_damage']}"
                )
    else:
        st.info("Move the time slider forward to see reports arrive...")

with tab2:
    buildings_with_reports = reports['building_id'].unique()
    
    if len(buildings_with_reports) > 0:
        selected_building = st.selectbox(
            "Select Building",
            buildings_with_reports,
            format_func=lambda x: f"Building {x}"
        )
        
        building_info = scenario[scenario['building_id'] == selected_building].iloc[0]
        building_beliefs = beliefs[beliefs['building_id'] == selected_building].iloc[0]
        building_reports = reports[reports['building_id'] == selected_building]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Building Information")
            st.metric("Type", building_info['building_type'].title())
            st.metric("Distance", f"{building_info['distance_km']:.1f} km")
            st.metric("Occupancy", f"{building_info['occupancy']} people")
            st.metric("True Damage", building_info['true_damage'].title())
            
            st.divider()
            
            st.subheader("Inference Results")
            st.metric(
                "P(Collapse)", 
                f"{building_beliefs['p_collapse']:.3f}"
            )
            st.metric("Std Dev", f"{building_beliefs['p_collapse_std']:.3f}")
        
            st.metric("Entropy", f"{building_beliefs['entropy']:.2f}")
            st.metric("Reports Received", int(building_beliefs['num_reports']))
        
        with col2:
            st.subheader("Belief Evolution")
            
            prior = np.array([
                building_info['p_none'],
                building_info['p_minor'],
                building_info['p_severe'],
                building_info['p_collapse']
            ])
            
            evolution = [{'time': 0, 'p_collapse': prior[3], 'event': 'Prior'}]
            current = prior.copy()
            
            from update_bayesian import bayesian_update
            from generate_reports import get_report_reliability
            
            for _, report in building_reports.iterrows():
                reliability = get_report_reliability(report['source'])
                current = bayesian_update(current, report['reported_state'], reliability)
                evolution.append({
                    'time': report['time_minutes'],
                    'p_collapse': current[3],
                    'event': f"{report['source']}: {report['reported_state']}"
                })
            
            evolution_df = pd.DataFrame(evolution)
            
            fig_evolution = go.Figure()
            
            fig_evolution.add_trace(go.Scatter(
                x=evolution_df['time'],
                y=evolution_df['p_collapse'],
                mode='lines+markers',
                name='P(Collapse)',
                line=dict(color='red', width=3),
                hovertemplate='%{text}<br>P(Collapse): %{y:.3f}<extra></extra>',
                text=evolution_df['event']
            ))
            
            fig_evolution.update_layout(
                title=f"Building {selected_building} - Belief Updates Over Time",
                xaxis_title="Time (minutes)",
                yaxis_title="P(Collapse)",
                yaxis=dict(range=[0, 1]),
                height=400
            )
            
            st.plotly_chart(fig_evolution, use_container_width=True)
            
            st.subheader("Report Timeline")
            for _, report in building_reports.iterrows():
                reliability = get_report_reliability(report['source'])
                st.text(
                    f"t={report['time_minutes']:6.1f}m | "
                    f"{report['source']:20s} ({reliability:.0%}) | "
                    f"Reports: {report['reported_state']:8s} | "
                    f"True: {report['true_damage']}"
                )
    else:
        st.info("No buildings received reports in this simulation. Try increasing Poisson rates.")

with tab3:
    st.header("Decision Quality Comparison")
    
    improvement = metrics['improvement']
    improvement_pct = metrics['improvement_pct']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Naive Approach",
            f"{metrics['naive_lives_saved']} people",
            help="Using only distance-based priors"
        )
    
    with col2:
        st.metric(
            "Bayesian Approach",
            f"{metrics['bayesian_lives_saved']} people",
            delta=f"+{improvement}",
            help="Using updated posterior beliefs"
        )
    
    with col3:
        st.metric(
            "Improvement",
            f"{improvement_pct:.1f}%",
            delta=f"+{improvement} lives"
        )
    
    st.divider()
    
    if improvement > 0:
        st.success(
            f"By incorporating uncertain information via Bayesian inference, "
            f"we can reach {improvement} more people ({improvement_pct:.1f}% improvement) "
            f"with the same {n_teams} rescue teams."
        )
    elif improvement < 0:
        st.warning(
            f"In this scenario, the naive approach performed slightly better. "
            f"This can happen when reports are very noisy or when the prior is already well-calibrated."
        )
    else:
        st.info("Both approaches performed equally in this scenario.")
    
    st.subheader("Performance Metrics")
    
    map_data_full = scenario.merge(beliefs, on='building_id', suffixes=('_prior', '_posterior'))
    
    buildings_with_reports = reports['building_id'].unique()
    comparison = map_data_full[map_data_full['building_id'].isin(buildings_with_reports)].copy()
    
    if len(comparison) > 0:
        comparison['true_collapsed'] = (comparison['true_damage'] == 'collapse').astype(int)
        comparison['naive_pred'] = (comparison['p_collapse_prior'] > 0.5).astype(int)
        comparison['bayesian_pred'] = (comparison['p_collapse_posterior'] > 0.5).astype(int)
        
        naive_acc = (comparison['true_collapsed'] == comparison['naive_pred']).mean()
        bayesian_acc = (comparison['true_collapsed'] == comparison['bayesian_pred']).mean()
    
        
        st.metric("Naive Accuracy", f"{naive_acc:.1%}")
        st.metric("Bayesian Accuracy", f"{bayesian_acc:.1%}", delta=f"{bayesian_acc - naive_acc:+.1%}")

st.divider()
