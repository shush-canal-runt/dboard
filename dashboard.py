import pandas as pd
import numpy as np
from flask import Flask, render_template_string
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats

app = Flask(__name__)

# Load and process data
df = pd.read_csv('data_set - Лист1.csv')

# Parse European number format (€1,39 -> 1.39)
def parse_euro(value):
    if pd.isna(value):
        return 0.0
    cleaned = str(value).replace('€', '').replace(',', '.').strip()
    try:
        return float(cleaned)
    except:
        return 0.0

df['Revenue'] = df['Revenue'].apply(parse_euro)
df['Spent'] = df['Spent'].apply(parse_euro)

# Rename Data column to Day for consistency
df = df.rename(columns={'Data': 'Day'})
df['Day'] = pd.to_datetime(df['Day'], format='%d.%m.%Y')

# Add week number
df['Week'] = df['Day'].dt.isocalendar().week
df['Year'] = df['Day'].dt.isocalendar().year
df['Week_Start'] = df['Day'] - pd.to_timedelta(df['Day'].dt.dayofweek, unit='D')

# Calculate total metrics
total_users = df['UserID'].nunique()
total_sessions = df['Sessions'].sum()
total_revenue = df['Revenue'].sum()
total_costs = df['Spent'].sum()
total_purchases = df['Transactions'].sum()
paying_users = df[df['Revenue'] > 0]['UserID'].nunique()
arpu = total_revenue / total_users
arppu = total_revenue / paying_users if paying_users > 0 else 0
roi = ((total_revenue - total_costs) / total_costs) * 100 if total_costs > 0 else 0

# Weekly aggregations
weekly_metrics = df.groupby('Week_Start').agg({
    'UserID': 'nunique',
    'Sessions': 'sum',
    'Revenue': 'sum',
    'Spent': 'sum',
    'Transactions': 'sum'
}).reset_index()
weekly_metrics.columns = ['Week_Start', 'Active_Users', 'Sessions', 'Revenue', 'Costs', 'Purchases']

# New users per week
first_appearance = df.groupby('UserID')['Week_Start'].min().reset_index()
first_appearance.columns = ['UserID', 'First_Week']
new_users_per_week = first_appearance.groupby('First_Week').size().reset_index(name='New_Users')
new_users_per_week.columns = ['Week_Start', 'New_Users']
weekly_metrics = weekly_metrics.merge(new_users_per_week, on='Week_Start', how='left')
weekly_metrics['New_Users'] = weekly_metrics['New_Users'].fillna(0).astype(int)
weekly_metrics['Cumulative_Users'] = weekly_metrics['New_Users'].cumsum()

# Paying users per week
paying_users_weekly = df[df['Revenue'] > 0].groupby('Week_Start')['UserID'].nunique().reset_index()
paying_users_weekly.columns = ['Week_Start', 'Paying_Users']
weekly_metrics = weekly_metrics.merge(paying_users_weekly, on='Week_Start', how='left')
weekly_metrics['Paying_Users'] = weekly_metrics['Paying_Users'].fillna(0).astype(int)

# Weekly metrics
weekly_metrics['ARPU'] = weekly_metrics['Revenue'] / weekly_metrics['Active_Users']
weekly_metrics['ARPPU'] = weekly_metrics.apply(lambda x: x['Revenue'] / x['Paying_Users'] if x['Paying_Users'] > 0 else 0, axis=1)
weekly_metrics['ROI'] = weekly_metrics.apply(lambda x: ((x['Revenue'] - x['Costs']) / x['Costs'] * 100) if x['Costs'] > 0 else 0, axis=1)

# Cumulative metrics
weekly_metrics['Cumulative_Sessions'] = weekly_metrics['Sessions'].cumsum()
weekly_metrics['Cumulative_Revenue'] = weekly_metrics['Revenue'].cumsum()
weekly_metrics['Cumulative_Costs'] = weekly_metrics['Costs'].cumsum()
weekly_metrics['Cumulative_Purchases'] = weekly_metrics['Purchases'].cumsum()
weekly_metrics['Cumulative_ARPU'] = weekly_metrics['Cumulative_Revenue'] / weekly_metrics['Cumulative_Users']
weekly_metrics['Cumulative_ROI'] = ((weekly_metrics['Cumulative_Revenue'] - weekly_metrics['Cumulative_Costs']) / weekly_metrics['Cumulative_Costs']) * 100

# Cumulative paying users
paying_users_by_week = df[df['Revenue'] > 0].groupby('Week_Start')['UserID'].apply(set).reset_index()
paying_users_by_week.columns = ['Week_Start', 'Paying_Users_Set']
all_weeks = weekly_metrics['Week_Start'].tolist()
cumulative_paying_users = []
seen_paying_users = set()
for week in all_weeks:
    week_paying = paying_users_by_week[paying_users_by_week['Week_Start'] == week]['Paying_Users_Set']
    if len(week_paying) > 0:
        seen_paying_users = seen_paying_users.union(week_paying.values[0])
    cumulative_paying_users.append(len(seen_paying_users))
weekly_metrics['Cumulative_Paying_Users'] = cumulative_paying_users
weekly_metrics['Cumulative_ARPPU'] = weekly_metrics['Cumulative_Revenue'] / weekly_metrics['Cumulative_Paying_Users']

# Week-over-week changes
weekly_metrics['Users_WoW'] = weekly_metrics['Active_Users'].pct_change() * 100
weekly_metrics['Sessions_WoW'] = weekly_metrics['Sessions'].pct_change() * 100
weekly_metrics['Revenue_WoW'] = weekly_metrics['Revenue'].pct_change() * 100
weekly_metrics['Costs_WoW'] = weekly_metrics['Costs'].pct_change() * 100
weekly_metrics['Purchases_WoW'] = weekly_metrics['Purchases'].pct_change() * 100
weekly_metrics['ARPU_WoW'] = weekly_metrics['ARPU'].pct_change() * 100
weekly_metrics['ARPPU_WoW'] = weekly_metrics['ARPPU'].pct_change() * 100

# Format week labels
weekly_metrics['Week_Label'] = weekly_metrics['Week_Start'].dt.strftime('Week %V\n(%b %d)')

# ============== ANOMALY DETECTION ==============

def detect_anomalies_zscore(series, threshold=2.0):
    """Detect anomalies using Z-score method"""
    if len(series) < 3:
        return pd.Series([False] * len(series), index=series.index)
    z_scores = np.abs(stats.zscore(series.fillna(0)))
    return z_scores > threshold

def detect_anomalies_iqr(series, multiplier=1.5):
    """Detect anomalies using IQR method"""
    if len(series) < 4:
        return pd.Series([False] * len(series), index=series.index)
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (series < lower_bound) | (series > upper_bound)

def detect_anomalies_wow_change(series, threshold=50):
    """Detect anomalies based on week-over-week percentage change"""
    pct_change = series.pct_change() * 100
    return np.abs(pct_change) > threshold

# Daily anomaly detection
daily_df = df.groupby('Day').agg({
    'UserID': 'nunique',
    'Sessions': 'sum',
    'Revenue': 'sum',
    'Spent': 'sum',
    'Transactions': 'sum'
}).reset_index()
daily_df.columns = ['Day', 'Active_Users', 'Sessions', 'Revenue', 'Costs', 'Purchases']

# Detect anomalies for each metric
anomalies = {}
metrics_to_check = ['Active_Users', 'Sessions', 'Revenue', 'Costs', 'Purchases']

for metric in metrics_to_check:
    anomalies[metric] = {
        'zscore': detect_anomalies_zscore(daily_df[metric]),
        'iqr': detect_anomalies_iqr(daily_df[metric]),
        'wow': detect_anomalies_wow_change(daily_df[metric])
    }

# Combine anomaly detection methods (flag if any method detects anomaly)
daily_df['Users_Anomaly'] = anomalies['Active_Users']['zscore'] | anomalies['Active_Users']['iqr']
daily_df['Sessions_Anomaly'] = anomalies['Sessions']['zscore'] | anomalies['Sessions']['iqr']
daily_df['Revenue_Anomaly'] = anomalies['Revenue']['zscore'] | anomalies['Revenue']['iqr']
daily_df['Costs_Anomaly'] = anomalies['Costs']['zscore'] | anomalies['Costs']['iqr']
daily_df['Purchases_Anomaly'] = anomalies['Purchases']['zscore'] | anomalies['Purchases']['iqr']

# Weekly anomaly detection
weekly_anomalies = {}
for metric in ['Active_Users', 'Sessions', 'Revenue', 'Costs', 'Purchases']:
    weekly_anomalies[metric] = {
        'zscore': detect_anomalies_zscore(weekly_metrics[metric]),
        'iqr': detect_anomalies_iqr(weekly_metrics[metric]),
        'wow': detect_anomalies_wow_change(weekly_metrics[metric])
    }

weekly_metrics['Users_Anomaly'] = weekly_anomalies['Active_Users']['zscore'] | weekly_anomalies['Active_Users']['wow']
weekly_metrics['Sessions_Anomaly'] = weekly_anomalies['Sessions']['zscore'] | weekly_anomalies['Sessions']['wow']
weekly_metrics['Revenue_Anomaly'] = weekly_anomalies['Revenue']['zscore'] | weekly_anomalies['Revenue']['wow']
weekly_metrics['Costs_Anomaly'] = weekly_anomalies['Costs']['zscore'] | weekly_anomalies['Costs']['wow']
weekly_metrics['Purchases_Anomaly'] = weekly_anomalies['Purchases']['zscore'] | weekly_anomalies['Purchases']['wow']

# User-level anomaly detection
user_metrics = df.groupby('UserID').agg({
    'Sessions': 'sum',
    'Revenue': 'sum',
    'Spent': 'sum',
    'Transactions': 'sum',
    'Duration': 'sum',
    'Day': 'nunique'
}).reset_index()
user_metrics.columns = ['UserID', 'Total_Sessions', 'Total_Revenue', 'Total_Spent', 'Total_Purchases', 'Total_Duration', 'Active_Days']
user_metrics['Avg_Session_Duration'] = user_metrics['Total_Duration'] / user_metrics['Total_Sessions']
user_metrics['Revenue_Per_Session'] = user_metrics['Total_Revenue'] / user_metrics['Total_Sessions']

# Detect user anomalies
user_metrics['Revenue_Anomaly'] = detect_anomalies_zscore(user_metrics['Total_Revenue'], threshold=2.5)
user_metrics['Sessions_Anomaly'] = detect_anomalies_zscore(user_metrics['Total_Sessions'], threshold=2.5)
user_metrics['Duration_Anomaly'] = detect_anomalies_zscore(user_metrics['Avg_Session_Duration'], threshold=2.5)
user_metrics['Spending_Anomaly'] = detect_anomalies_zscore(user_metrics['Total_Spent'], threshold=2.5)

# Get anomalous users
anomalous_users = user_metrics[
    user_metrics['Revenue_Anomaly'] | 
    user_metrics['Sessions_Anomaly'] | 
    user_metrics['Duration_Anomaly'] |
    user_metrics['Spending_Anomaly']
].copy()

# Collect all anomalies for summary
def get_anomaly_summary():
    summary = []
    
    # Daily anomalies
    for idx, row in daily_df.iterrows():
        anomaly_types = []
        if row.get('Users_Anomaly', False):
            anomaly_types.append(f"Users: {row['Active_Users']}")
        if row.get('Sessions_Anomaly', False):
            anomaly_types.append(f"Sessions: {row['Sessions']}")
        if row.get('Revenue_Anomaly', False):
            anomaly_types.append(f"Revenue: €{row['Revenue']:.2f}")
        if row.get('Costs_Anomaly', False):
            anomaly_types.append(f"Costs: €{row['Costs']:.2f}")
        if row.get('Purchases_Anomaly', False):
            anomaly_types.append(f"Purchases: {row['Purchases']}")
        
        if anomaly_types:
            summary.append({
                'type': 'Daily',
                'date': row['Day'].strftime('%Y-%m-%d'),
                'description': ', '.join(anomaly_types),
                'severity': 'High' if len(anomaly_types) > 2 else 'Medium' if len(anomaly_types) > 1 else 'Low'
            })
    
    # Weekly anomalies
    for idx, row in weekly_metrics.iterrows():
        anomaly_types = []
        if row.get('Users_Anomaly', False):
            anomaly_types.append(f"Users: {row['Active_Users']}")
        if row.get('Sessions_Anomaly', False):
            anomaly_types.append(f"Sessions: {row['Sessions']}")
        if row.get('Revenue_Anomaly', False):
            anomaly_types.append(f"Revenue: €{row['Revenue']:.2f}")
        if row.get('Costs_Anomaly', False):
            anomaly_types.append(f"Costs: €{row['Costs']:.2f}")
        if row.get('Purchases_Anomaly', False):
            anomaly_types.append(f"Purchases: {row['Purchases']}")
        
        if anomaly_types:
            week_label = f"{row['Week_Start'].strftime('%b %d')} - {(row['Week_Start'] + pd.Timedelta(days=6)).strftime('%b %d')}"
            summary.append({
                'type': 'Weekly',
                'date': week_label,
                'description': ', '.join(anomaly_types),
                'severity': 'High' if len(anomaly_types) > 2 else 'Medium' if len(anomaly_types) > 1 else 'Low'
            })
    
    # User anomalies
    for idx, row in anomalous_users.iterrows():
        anomaly_types = []
        if row.get('Revenue_Anomaly', False):
            anomaly_types.append(f"Revenue: €{row['Total_Revenue']:.2f}")
        if row.get('Sessions_Anomaly', False):
            anomaly_types.append(f"Sessions: {row['Total_Sessions']}")
        if row.get('Duration_Anomaly', False):
            anomaly_types.append(f"Avg Duration: {row['Avg_Session_Duration']:.1f}s")
        if row.get('Spending_Anomaly', False):
            anomaly_types.append(f"Acquisition Cost: €{row['Total_Spent']:.2f}")
        
        if anomaly_types:
            summary.append({
                'type': 'User',
                'date': f"UserID: {row['UserID']}",
                'description': ', '.join(anomaly_types),
                'severity': 'High' if len(anomaly_types) > 2 else 'Medium' if len(anomaly_types) > 1 else 'Low'
            })
    
    return summary

anomaly_summary = get_anomaly_summary()

# ============== COHORT ANALYSIS ==============

# Get first activity date for each user (cohort assignment)
user_first_activity = df.groupby('UserID')['Day'].min().reset_index()
user_first_activity.columns = ['UserID', 'First_Activity']
user_first_activity['Cohort'] = user_first_activity['First_Activity'].dt.to_period('W').dt.start_time

# Merge cohort info back to main dataframe
df_cohort = df.merge(user_first_activity[['UserID', 'Cohort']], on='UserID')

# Calculate period number (weeks since first activity)
df_cohort['Period'] = ((df_cohort['Day'] - df_cohort['Cohort']).dt.days // 7).astype(int)

# Retention Cohort Analysis
cohort_retention = df_cohort.groupby(['Cohort', 'Period'])['UserID'].nunique().reset_index()
cohort_retention.columns = ['Cohort', 'Period', 'Users']

# Pivot for retention matrix
retention_matrix = cohort_retention.pivot(index='Cohort', columns='Period', values='Users')

# Calculate cohort sizes (users in period 0)
cohort_sizes = retention_matrix[0].copy()

# Calculate retention percentages
retention_pct = retention_matrix.divide(cohort_sizes, axis=0) * 100

# Revenue Cohort Analysis
cohort_revenue = df_cohort.groupby(['Cohort', 'Period'])['Revenue'].sum().reset_index()
cohort_revenue.columns = ['Cohort', 'Period', 'Revenue']
revenue_matrix = cohort_revenue.pivot(index='Cohort', columns='Period', values='Revenue')

# Cumulative Revenue per Cohort
cumulative_revenue_matrix = revenue_matrix.cumsum(axis=1)

# LTV per user in cohort
ltv_matrix = cumulative_revenue_matrix.divide(cohort_sizes, axis=0)

# Sessions Cohort Analysis
cohort_sessions = df_cohort.groupby(['Cohort', 'Period'])['Sessions'].sum().reset_index()
cohort_sessions.columns = ['Cohort', 'Period', 'Sessions']
sessions_matrix = cohort_sessions.pivot(index='Cohort', columns='Period', values='Sessions')

# Purchases Cohort Analysis
cohort_purchases = df_cohort.groupby(['Cohort', 'Period'])['Transactions'].sum().reset_index()
cohort_purchases.columns = ['Cohort', 'Period', 'Purchases']
purchases_matrix = cohort_purchases.pivot(index='Cohort', columns='Period', values='Purchases')

# Format cohort labels
cohort_labels = [c.strftime('%b %d') for c in retention_matrix.index]

def create_cohort_heatmap(matrix, title, colorscale='Blues', fmt='.1f', suffix='%'):
    """Create a heatmap for cohort analysis"""
    # Prepare data
    z_data = matrix.values
    x_labels = [f'Week {i}' for i in matrix.columns]
    y_labels = [c.strftime('%b %d') for c in matrix.index]
    
    # Create text annotations
    text_data = []
    for row in z_data:
        text_row = []
        for val in row:
            if pd.isna(val):
                text_row.append('')
            else:
                text_row.append(f'{val:{fmt}}{suffix}')
        text_data.append(text_row)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        text=text_data,
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Weeks Since First Activity',
        yaxis_title='Cohort (Week of First Activity)',
        height=400,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

def create_cohort_charts():
    """Create all cohort analysis charts"""
    # Create subplots for cohort analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'User Retention by Cohort (%)',
            'Cumulative Revenue per User (LTV)',
            'Weekly Revenue by Cohort',
            'Weekly Sessions by Cohort'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Prepare data
    x_labels = [f'Week {i}' for i in retention_pct.columns]
    y_labels = [c.strftime('%b %d') for c in retention_pct.index]
    
    # 1. Retention Heatmap
    z_retention = retention_pct.values
    text_retention = [[f'{v:.1f}%' if not pd.isna(v) else '' for v in row] for row in z_retention]
    
    fig.add_trace(go.Heatmap(
        z=z_retention,
        x=x_labels,
        y=y_labels,
        colorscale='Blues',
        text=text_retention,
        texttemplate='%{text}',
        textfont={"size": 9},
        showscale=False,
        hoverongaps=False
    ), row=1, col=1)
    
    # 2. LTV Heatmap
    z_ltv = ltv_matrix.values
    text_ltv = [[f'€{v:.2f}' if not pd.isna(v) else '' for v in row] for row in z_ltv]
    
    fig.add_trace(go.Heatmap(
        z=z_ltv,
        x=x_labels,
        y=y_labels,
        colorscale='Greens',
        text=text_ltv,
        texttemplate='%{text}',
        textfont={"size": 9},
        showscale=False,
        hoverongaps=False
    ), row=1, col=2)
    
    # 3. Revenue Heatmap
    z_revenue = revenue_matrix.values
    text_revenue = [[f'€{v:.0f}' if not pd.isna(v) else '' for v in row] for row in z_revenue]
    
    fig.add_trace(go.Heatmap(
        z=z_revenue,
        x=x_labels,
        y=y_labels,
        colorscale='Oranges',
        text=text_revenue,
        texttemplate='%{text}',
        textfont={"size": 9},
        showscale=False,
        hoverongaps=False
    ), row=2, col=1)
    
    # 4. Sessions Heatmap
    z_sessions = sessions_matrix.values
    text_sessions = [[f'{int(v)}' if not pd.isna(v) else '' for v in row] for row in z_sessions]
    
    fig.add_trace(go.Heatmap(
        z=z_sessions,
        x=x_labels,
        y=y_labels,
        colorscale='Purples',
        text=text_sessions,
        texttemplate='%{text}',
        textfont={"size": 9},
        showscale=False,
        hoverongaps=False
    ), row=2, col=2)
    
    fig.update_layout(
        height=700,
        title_text="Cohort Analysis",
        title_x=0.5,
        title_font_size=20,
        paper_bgcolor='#f8f9fa',
        plot_bgcolor='white'
    )
    
    return fig.to_html(full_html=False)

# Calculate cohort summary statistics
cohort_summary = []
for cohort in retention_matrix.index:
    cohort_size = cohort_sizes[cohort]
    total_rev = revenue_matrix.loc[cohort].sum()
    avg_ltv = total_rev / cohort_size if cohort_size > 0 else 0
    
    # Calculate retention for week 1 if available
    week1_retention = retention_pct.loc[cohort, 1] if 1 in retention_pct.columns and not pd.isna(retention_pct.loc[cohort, 1]) else 0
    
    # Calculate average retention across all periods
    avg_retention = retention_pct.loc[cohort].mean()
    
    cohort_summary.append({
        'cohort': cohort.strftime('%b %d'),
        'size': int(cohort_size),
        'total_revenue': total_rev,
        'avg_ltv': avg_ltv,
        'week1_retention': week1_retention,
        'avg_retention': avg_retention
    })

def create_anomaly_charts():
    """Create anomaly detection visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Daily Revenue with Anomalies',
            'Daily Sessions with Anomalies',
            'User Revenue Distribution',
            'User Sessions Distribution'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Daily Revenue with anomalies
    anomaly_mask = daily_df['Revenue_Anomaly']
    fig.add_trace(go.Scatter(
        x=daily_df['Day'], y=daily_df['Revenue'],
        mode='lines+markers',
        name='Revenue',
        line=dict(color='#44AF69', width=2),
        marker=dict(size=6)
    ), row=1, col=1)
    
    if anomaly_mask.any():
        fig.add_trace(go.Scatter(
            x=daily_df[anomaly_mask]['Day'],
            y=daily_df[anomaly_mask]['Revenue'],
            mode='markers',
            name='Revenue Anomaly',
            marker=dict(color='red', size=14, symbol='x', line=dict(width=2))
        ), row=1, col=1)
    
    # Daily Sessions with anomalies
    anomaly_mask = daily_df['Sessions_Anomaly']
    fig.add_trace(go.Scatter(
        x=daily_df['Day'], y=daily_df['Sessions'],
        mode='lines+markers',
        name='Sessions',
        line=dict(color='#F18F01', width=2),
        marker=dict(size=6)
    ), row=1, col=2)
    
    if anomaly_mask.any():
        fig.add_trace(go.Scatter(
            x=daily_df[anomaly_mask]['Day'],
            y=daily_df[anomaly_mask]['Sessions'],
            mode='markers',
            name='Sessions Anomaly',
            marker=dict(color='red', size=14, symbol='x', line=dict(width=2))
        ), row=1, col=2)
    
    # User Revenue Distribution (Box plot)
    fig.add_trace(go.Box(
        y=user_metrics['Total_Revenue'],
        name='Revenue',
        boxpoints='outliers',
        marker_color='#44AF69',
        line_color='#44AF69'
    ), row=2, col=1)
    
    # Highlight anomalous users
    if user_metrics['Revenue_Anomaly'].any():
        anomalous = user_metrics[user_metrics['Revenue_Anomaly']]
        fig.add_trace(go.Scatter(
            x=['Revenue'] * len(anomalous),
            y=anomalous['Total_Revenue'],
            mode='markers',
            name='Anomalous Users (Revenue)',
            marker=dict(color='red', size=12, symbol='diamond')
        ), row=2, col=1)
    
    # User Sessions Distribution (Box plot)
    fig.add_trace(go.Box(
        y=user_metrics['Total_Sessions'],
        name='Sessions',
        boxpoints='outliers',
        marker_color='#F18F01',
        line_color='#F18F01'
    ), row=2, col=2)
    
    # Highlight anomalous users
    if user_metrics['Sessions_Anomaly'].any():
        anomalous = user_metrics[user_metrics['Sessions_Anomaly']]
        fig.add_trace(go.Scatter(
            x=['Sessions'] * len(anomalous),
            y=anomalous['Total_Sessions'],
            mode='markers',
            name='Anomalous Users (Sessions)',
            marker=dict(color='red', size=12, symbol='diamond')
        ), row=2, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Anomaly Detection Analysis",
        title_x=0.5,
        title_font_size=20,
        paper_bgcolor='#f8f9fa',
        plot_bgcolor='white'
    )
    
    return fig.to_html(full_html=False)

def create_dashboard():
    # Create subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Users (Cumulative & Weekly Active)',
            'Sessions (Cumulative & Weekly)',
            'Revenue (Cumulative & Weekly)',
            'Costs (Cumulative & Weekly)',
            'Purchases (Cumulative & Weekly)',
            'ARPU (Cumulative & Weekly)',
            'ARPPU (Cumulative & Weekly)',
            'ROI (Cumulative & Weekly)'
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )
    
    # 1. Users
    fig.add_trace(go.Scatter(x=weekly_metrics['Week_Label'], y=weekly_metrics['Cumulative_Users'], 
                             name='Cumulative Users', line=dict(color='#2E86AB', width=3),
                             mode='lines+markers', marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Bar(x=weekly_metrics['Week_Label'], y=weekly_metrics['Active_Users'], 
                         name='Weekly Active Users', marker_color='#A23B72', opacity=0.7,
                         text=weekly_metrics['Users_WoW'].apply(lambda x: f'{x:+.1f}%' if pd.notna(x) else ''),
                         textposition='outside'), row=1, col=1)
    
    # 2. Sessions
    fig.add_trace(go.Scatter(x=weekly_metrics['Week_Label'], y=weekly_metrics['Cumulative_Sessions'], 
                             name='Cumulative Sessions', line=dict(color='#2E86AB', width=3),
                             mode='lines+markers', marker=dict(size=8)), row=1, col=2)
    fig.add_trace(go.Bar(x=weekly_metrics['Week_Label'], y=weekly_metrics['Sessions'], 
                         name='Weekly Sessions', marker_color='#F18F01', opacity=0.7,
                         text=weekly_metrics['Sessions_WoW'].apply(lambda x: f'{x:+.1f}%' if pd.notna(x) else ''),
                         textposition='outside'), row=1, col=2)
    
    # 3. Revenue
    fig.add_trace(go.Scatter(x=weekly_metrics['Week_Label'], y=weekly_metrics['Cumulative_Revenue'], 
                             name='Cumulative Revenue', line=dict(color='#2E86AB', width=3),
                             mode='lines+markers', marker=dict(size=8)), row=2, col=1)
    fig.add_trace(go.Bar(x=weekly_metrics['Week_Label'], y=weekly_metrics['Revenue'], 
                         name='Weekly Revenue', marker_color='#44AF69', opacity=0.7,
                         text=weekly_metrics['Revenue_WoW'].apply(lambda x: f'{x:+.1f}%' if pd.notna(x) else ''),
                         textposition='outside'), row=2, col=1)
    
    # 4. Costs
    fig.add_trace(go.Scatter(x=weekly_metrics['Week_Label'], y=weekly_metrics['Cumulative_Costs'], 
                             name='Cumulative Costs', line=dict(color='#2E86AB', width=3),
                             mode='lines+markers', marker=dict(size=8)), row=2, col=2)
    fig.add_trace(go.Bar(x=weekly_metrics['Week_Label'], y=weekly_metrics['Costs'], 
                         name='Weekly Costs', marker_color='#E63946', opacity=0.7,
                         text=weekly_metrics['Costs_WoW'].apply(lambda x: f'{x:+.1f}%' if pd.notna(x) else ''),
                         textposition='outside'), row=2, col=2)
    
    # 5. Purchases
    fig.add_trace(go.Scatter(x=weekly_metrics['Week_Label'], y=weekly_metrics['Cumulative_Purchases'], 
                             name='Cumulative Purchases', line=dict(color='#2E86AB', width=3),
                             mode='lines+markers', marker=dict(size=8)), row=3, col=1)
    fig.add_trace(go.Bar(x=weekly_metrics['Week_Label'], y=weekly_metrics['Purchases'], 
                         name='Weekly Purchases', marker_color='#9B5DE5', opacity=0.7,
                         text=weekly_metrics['Purchases_WoW'].apply(lambda x: f'{x:+.1f}%' if pd.notna(x) else ''),
                         textposition='outside'), row=3, col=1)
    
    # 6. ARPU
    fig.add_trace(go.Scatter(x=weekly_metrics['Week_Label'], y=weekly_metrics['Cumulative_ARPU'], 
                             name='Cumulative ARPU', line=dict(color='#2E86AB', width=3),
                             mode='lines+markers', marker=dict(size=8)), row=3, col=2)
    fig.add_trace(go.Bar(x=weekly_metrics['Week_Label'], y=weekly_metrics['ARPU'], 
                         name='Weekly ARPU', marker_color='#00BBF9', opacity=0.7,
                         text=weekly_metrics['ARPU_WoW'].apply(lambda x: f'{x:+.1f}%' if pd.notna(x) else ''),
                         textposition='outside'), row=3, col=2)
    
    # 7. ARPPU
    fig.add_trace(go.Scatter(x=weekly_metrics['Week_Label'], y=weekly_metrics['Cumulative_ARPPU'], 
                             name='Cumulative ARPPU', line=dict(color='#2E86AB', width=3),
                             mode='lines+markers', marker=dict(size=8)), row=4, col=1)
    fig.add_trace(go.Bar(x=weekly_metrics['Week_Label'], y=weekly_metrics['ARPPU'], 
                         name='Weekly ARPPU', marker_color='#F15BB5', opacity=0.7,
                         text=weekly_metrics['ARPPU_WoW'].apply(lambda x: f'{x:+.1f}%' if pd.notna(x) else ''),
                         textposition='outside'), row=4, col=1)
    
    # 8. ROI
    fig.add_trace(go.Scatter(x=weekly_metrics['Week_Label'], y=weekly_metrics['Cumulative_ROI'], 
                             name='Cumulative ROI', line=dict(color='#2E86AB', width=3),
                             mode='lines+markers', marker=dict(size=8)), row=4, col=2)
    fig.add_trace(go.Bar(x=weekly_metrics['Week_Label'], y=weekly_metrics['ROI'], 
                         name='Weekly ROI', marker_color='#FEE440', opacity=0.7), row=4, col=2)
    
    fig.update_layout(
        height=1400,
        showlegend=False,
        title_text="Mobile App Analytics Dashboard - Week-to-Week Dynamics",
        title_x=0.5,
        title_font_size=24,
        paper_bgcolor='#f8f9fa',
        plot_bgcolor='white'
    )
    
    # Update axes
    for i in range(1, 5):
        for j in range(1, 3):
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0', row=i, col=j)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0', row=i, col=j)
    
    return fig.to_html(full_html=False)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Mobile App Analytics Dashboard - Weekly</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #2E86AB;
            margin-bottom: 10px;
        }
        .header p {
            color: #666;
            font-size: 14px;
        }
        .kpi-container {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }
        .kpi-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .kpi-card h3 {
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
        }
        .kpi-card .value {
            font-size: 28px;
            font-weight: bold;
            color: #2E86AB;
        }
        .kpi-card .subtext {
            font-size: 12px;
            color: #999;
            margin-top: 5px;
        }
        .kpi-card .wow {
            font-size: 14px;
            margin-top: 8px;
            padding: 4px 8px;
            border-radius: 4px;
            display: inline-block;
        }
        .kpi-card .wow.positive {
            background-color: #d4edda;
            color: #155724;
        }
        .kpi-card .wow.negative {
            background-color: #f8d7da;
            color: #721c24;
        }
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .weekly-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .weekly-table th, .weekly-table td {
            padding: 12px 15px;
            text-align: center;
            border-bottom: 1px solid #e0e0e0;
        }
        .weekly-table th {
            background-color: #2E86AB;
            color: white;
            font-weight: 600;
        }
        .weekly-table tr:hover {
            background-color: #f5f5f5;
        }
        .positive { color: #155724; }
        .negative { color: #721c24; }
        
        /* Anomaly Section Styles */
        .section-title {
            color: #2E86AB;
            font-size: 24px;
            margin: 40px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #2E86AB;
        }
        .anomaly-summary {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }
        .anomaly-stat {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .anomaly-stat.high {
            border-left: 4px solid #dc3545;
        }
        .anomaly-stat.medium {
            border-left: 4px solid #ffc107;
        }
        .anomaly-stat.low {
            border-left: 4px solid #28a745;
        }
        .anomaly-stat h4 {
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
        }
        .anomaly-stat .count {
            font-size: 36px;
            font-weight: bold;
        }
        .anomaly-stat.high .count { color: #dc3545; }
        .anomaly-stat.medium .count { color: #ffc107; }
        .anomaly-stat.low .count { color: #28a745; }
        
        .anomaly-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .anomaly-table th, .anomaly-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        .anomaly-table th {
            background-color: #dc3545;
            color: white;
            font-weight: 600;
        }
        .anomaly-table tr:hover {
            background-color: #fff5f5;
        }
        .severity-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .severity-high {
            background-color: #f8d7da;
            color: #721c24;
        }
        .severity-medium {
            background-color: #fff3cd;
            color: #856404;
        }
        .severity-low {
            background-color: #d4edda;
            color: #155724;
        }
        .type-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .type-daily {
            background-color: #cce5ff;
            color: #004085;
        }
        .type-weekly {
            background-color: #d4edda;
            color: #155724;
        }
        .type-user {
            background-color: #e2d5f1;
            color: #5a3d8a;
        }
        
        /* Cohort Analysis Styles */
        .cohort-summary {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }
        .cohort-stat {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            border-left: 4px solid #2E86AB;
        }
        .cohort-stat h4 {
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
        }
        .cohort-stat .count {
            font-size: 32px;
            font-weight: bold;
            color: #2E86AB;
        }
        .cohort-stat .subtext {
            font-size: 12px;
            color: #999;
            margin-top: 5px;
        }
        .cohort-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .cohort-table th, .cohort-table td {
            padding: 12px 15px;
            text-align: center;
            border-bottom: 1px solid #e0e0e0;
        }
        .cohort-table th {
            background-color: #2E86AB;
            color: white;
            font-weight: 600;
        }
        .cohort-table tr:hover {
            background-color: #f5f5f5;
        }
        
        @media (max-width: 1200px) {
            .kpi-container {
                grid-template-columns: repeat(2, 1fr);
            }
            .anomaly-summary {
                grid-template-columns: repeat(1, 1fr);
            }
            .cohort-summary {
                grid-template-columns: repeat(1, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📱 Mobile App Analytics Dashboard</h1>
        <p>Data Period: {{ date_range }} | Week-to-Week Dynamics</p>
    </div>
    
    <div class="kpi-container">
        <div class="kpi-card">
            <h3>Total Users</h3>
            <div class="value">{{ total_users }}</div>
            <div class="subtext">Paying: {{ paying_users }} ({{ paying_pct }}%)</div>
        </div>
        <div class="kpi-card">
            <h3>Total Sessions</h3>
            <div class="value">{{ total_sessions }}</div>
            <div class="subtext">Weekly avg: {{ weekly_avg_sessions }}</div>
        </div>
        <div class="kpi-card">
            <h3>Total Revenue</h3>
            <div class="value">€{{ total_revenue }}</div>
            <div class="subtext">Weekly avg: €{{ weekly_avg_revenue }}</div>
        </div>
        <div class="kpi-card">
            <h3>Total Costs</h3>
            <div class="value">€{{ total_costs }}</div>
            <div class="subtext">Weekly avg: €{{ weekly_avg_costs }}</div>
        </div>
        <div class="kpi-card">
            <h3>Total Purchases</h3>
            <div class="value">{{ total_purchases }}</div>
            <div class="subtext">Avg per paying user: {{ avg_purchases }}</div>
        </div>
        <div class="kpi-card">
            <h3>ARPU</h3>
            <div class="value">€{{ arpu }}</div>
            <div class="subtext">Revenue per user</div>
        </div>
        <div class="kpi-card">
            <h3>ARPPU</h3>
            <div class="value">€{{ arppu }}</div>
            <div class="subtext">Revenue per paying user</div>
        </div>
        <div class="kpi-card">
            <h3>ROI</h3>
            <div class="value" style="color: {{ roi_color }}">{{ roi }}%</div>
            <div class="subtext">Return on Investment</div>
        </div>
    </div>
    
    <div class="chart-container">
        {{ chart | safe }}
    </div>
    
    <table class="weekly-table">
        <thead>
            <tr>
                <th>Week</th>
                <th>Active Users</th>
                <th>Sessions</th>
                <th>Revenue</th>
                <th>Costs</th>
                <th>Purchases</th>
                <th>ARPU</th>
                <th>ARPPU</th>
                <th>ROI</th>
            </tr>
        </thead>
        <tbody>
            {{ weekly_table | safe }}
        </tbody>
    </table>
    
    <!-- Anomaly Detection Section -->
    <h2 class="section-title">🔍 Anomaly Detection</h2>
    
    <div class="anomaly-summary">
        <div class="anomaly-stat high">
            <h4>High Severity</h4>
            <div class="count">{{ anomaly_high }}</div>
            <div class="subtext">Critical anomalies</div>
        </div>
        <div class="anomaly-stat medium">
            <h4>Medium Severity</h4>
            <div class="count">{{ anomaly_medium }}</div>
            <div class="subtext">Warning anomalies</div>
        </div>
        <div class="anomaly-stat low">
            <h4>Low Severity</h4>
            <div class="count">{{ anomaly_low }}</div>
            <div class="subtext">Minor anomalies</div>
        </div>
    </div>
    
    <div class="chart-container">
        {{ anomaly_chart | safe }}
    </div>
    
    <!-- Cohort Analysis Section -->
    <h2 class="section-title">📊 Cohort Analysis</h2>
    
    <div class="cohort-summary">
        <div class="cohort-stat">
            <h4>Total Cohorts</h4>
            <div class="count">{{ total_cohorts }}</div>
            <div class="subtext">Weekly cohorts analyzed</div>
        </div>
        <div class="cohort-stat">
            <h4>Avg Week 1 Retention</h4>
            <div class="count">{{ avg_week1_retention }}%</div>
            <div class="subtext">Users returning after 1 week</div>
        </div>
        <div class="cohort-stat">
            <h4>Avg LTV</h4>
            <div class="count">€{{ avg_ltv }}</div>
            <div class="subtext">Lifetime value per user</div>
        </div>
    </div>
    
    <div class="chart-container">
        {{ cohort_chart | safe }}
    </div>
    
    <table class="cohort-table">
        <thead>
            <tr>
                <th>Cohort</th>
                <th>Size</th>
                <th>Total Revenue</th>
                <th>Avg LTV</th>
                <th>Week 1 Retention</th>
                <th>Avg Retention</th>
            </tr>
        </thead>
        <tbody>
            {{ cohort_table | safe }}
        </tbody>
    </table>
</body>
</html>
'''

def generate_weekly_table():
    rows = []
    for _, row in weekly_metrics.iterrows():
        wow_users = f'<span class="{"positive" if row["Users_WoW"] >= 0 else "negative"}">{row["Users_WoW"]:+.1f}%</span>' if pd.notna(row['Users_WoW']) else '-'
        wow_sessions = f'<span class="{"positive" if row["Sessions_WoW"] >= 0 else "negative"}">{row["Sessions_WoW"]:+.1f}%</span>' if pd.notna(row['Sessions_WoW']) else '-'
        wow_revenue = f'<span class="{"positive" if row["Revenue_WoW"] >= 0 else "negative"}">{row["Revenue_WoW"]:+.1f}%</span>' if pd.notna(row['Revenue_WoW']) else '-'
        wow_costs = f'<span class="{"negative" if row["Costs_WoW"] >= 0 else "positive"}">{row["Costs_WoW"]:+.1f}%</span>' if pd.notna(row['Costs_WoW']) else '-'
        wow_purchases = f'<span class="{"positive" if row["Purchases_WoW"] >= 0 else "negative"}">{row["Purchases_WoW"]:+.1f}%</span>' if pd.notna(row['Purchases_WoW']) else '-'
        wow_arpu = f'<span class="{"positive" if row["ARPU_WoW"] >= 0 else "negative"}">{row["ARPU_WoW"]:+.1f}%</span>' if pd.notna(row['ARPU_WoW']) else '-'
        wow_arppu = f'<span class="{"positive" if row["ARPPU_WoW"] >= 0 else "negative"}">{row["ARPPU_WoW"]:+.1f}%</span>' if pd.notna(row['ARPPU_WoW']) else '-'
        
        rows.append(f'''
            <tr>
                <td>{row['Week_Start'].strftime('%b %d')} - {(row['Week_Start'] + pd.Timedelta(days=6)).strftime('%b %d')}</td>
                <td>{row['Active_Users']}<br><small>{wow_users}</small></td>
                <td>{row['Sessions']}<br><small>{wow_sessions}</small></td>
                <td>€{row['Revenue']:.2f}<br><small>{wow_revenue}</small></td>
                <td>€{row['Costs']:.2f}<br><small>{wow_costs}</small></td>
                <td>{row['Purchases']}<br><small>{wow_purchases}</small></td>
                <td>€{row['ARPU']:.2f}<br><small>{wow_arpu}</small></td>
                <td>€{row['ARPPU']:.2f}<br><small>{wow_arppu}</small></td>
                <td><span class="{"positive" if row['ROI'] > 0 else "negative"}">{row['ROI']:.1f}%</span></td>
            </tr>
        ''')
    return ''.join(rows)

def generate_anomaly_table():
    rows = []
    for anomaly in anomaly_summary:
        type_class = f"type-{anomaly['type'].lower()}"
        severity_class = f"severity-{anomaly['severity'].lower()}"
        rows.append(f'''
            <tr>
                <td><span class="type-badge {type_class}">{anomaly['type']}</span></td>
                <td>{anomaly['date']}</td>
                <td>{anomaly['description']}</td>
                <td><span class="severity-badge {severity_class}">{anomaly['severity']}</span></td>
            </tr>
        ''')
    return ''.join(rows) if rows else '<tr><td colspan="4" style="text-align:center;">No anomalies detected</td></tr>'

def generate_cohort_table():
    rows = []
    for cohort in cohort_summary:
        rows.append(f'''
            <tr>
                <td>{cohort['cohort']}</td>
                <td>{cohort['size']}</td>
                <td>€{cohort['total_revenue']:.2f}</td>
                <td>€{cohort['avg_ltv']:.2f}</td>
                <td>{cohort['week1_retention']:.1f}%</td>
                <td>{cohort['avg_retention']:.1f}%</td>
            </tr>
        ''')
    return ''.join(rows)

@app.route('/')
def dashboard():
    chart_html = create_dashboard()
    anomaly_chart_html = create_anomaly_charts()
    cohort_chart_html = create_cohort_charts()
    
    date_range = f"{df['Day'].min().strftime('%Y-%m-%d')} to {df['Day'].max().strftime('%Y-%m-%d')}"
    num_weeks = len(weekly_metrics)
    
    # Count anomalies by severity
    anomaly_high = sum(1 for a in anomaly_summary if a['severity'] == 'High')
    anomaly_medium = sum(1 for a in anomaly_summary if a['severity'] == 'Medium')
    anomaly_low = sum(1 for a in anomaly_summary if a['severity'] == 'Low')
    
    # Cohort statistics
    total_cohorts = len(cohort_summary)
    avg_week1_ret = np.mean([c['week1_retention'] for c in cohort_summary if c['week1_retention'] > 0])
    avg_ltv_val = np.mean([c['avg_ltv'] for c in cohort_summary])
    
    return render_template_string(
        HTML_TEMPLATE,
        chart=chart_html,
        anomaly_chart=anomaly_chart_html,
        cohort_chart=cohort_chart_html,
        date_range=date_range,
        total_users=total_users,
        paying_users=paying_users,
        paying_pct=round(paying_users/total_users*100, 1),
        total_sessions=f"{total_sessions:,}",
        weekly_avg_sessions=round(total_sessions/num_weeks, 1),
        total_revenue=f"{total_revenue:,.2f}",
        weekly_avg_revenue=f"{total_revenue/num_weeks:,.2f}",
        total_costs=f"{total_costs:,.2f}",
        weekly_avg_costs=f"{total_costs/num_weeks:,.2f}",
        total_purchases=total_purchases,
        avg_purchases=round(total_purchases/paying_users, 1),
        arpu=f"{arpu:.2f}",
        arppu=f"{arppu:.2f}",
        roi=f"{roi:.2f}",
        roi_color='#44AF69' if roi > 0 else '#E63946',
        weekly_table=generate_weekly_table(),
        anomaly_table=generate_anomaly_table(),
        anomaly_high=anomaly_high,
        anomaly_medium=anomaly_medium,
        anomaly_low=anomaly_low,
        cohort_table=generate_cohort_table(),
        total_cohorts=total_cohorts,
        avg_week1_retention=f"{avg_week1_ret:.1f}" if not np.isnan(avg_week1_ret) else "N/A",
        avg_ltv=f"{avg_ltv_val:.2f}" if not np.isnan(avg_ltv_val) else "N/A"
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12000, debug=False)

