"""Reusable UI components for the log visualizer."""

from typing import Any, Dict, Optional
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc

from .styles import (
    COLORS, FONTS, UPLOAD_STYLE, JSON_VIEWER_STYLE,
    HEADER_STYLE, TITLE_STYLE, SUBTITLE_STYLE,
    get_node_color, get_event_color
)


def create_header() -> dbc.Container:
    """Create the main header component."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Robotics Pipeline Analyzer", style=TITLE_STYLE),
                html.P(
                    "Interactive visualization and analysis of your robotics pipeline execution logs",
                    style=SUBTITLE_STYLE
                )
            ], width=12)
        ])
    ], style=HEADER_STYLE)


def create_upload_component() -> dbc.Card:
    """Create the file upload component."""
    return dbc.Card([
        dbc.CardBody([
            html.H4("Upload Log File", className="card-title mb-3"),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.I(className="fas fa-cloud-upload-alt fa-2x mb-2"),
                    html.Br(),
                    'Drag and Drop or ',
                    html.A('Select Log File', style={'color': COLORS['primary'], 'fontWeight': '600'})
                ]),
                style=UPLOAD_STYLE,
                multiple=False
            )
        ], style={'height': '300px'})  # Set card-body height to 300px
    ], style={'marginBottom': '24px'})


def create_stats_cards(stats: Optional[Dict[str, Any]] = None) -> dbc.Row:
    """Create statistics cards."""
    if not stats:
        return dbc.Row([])

    node_stats = stats.get('node_stats', {})

    cards = [
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(stats.get('total_logs', 0), className="card-title text-primary"),
                    html.P("Total Log Entries", className="card-text text-muted")
                ])
            ], color="primary", outline=True)
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(node_stats.get('total_nodes', 0), className="card-title text-success"),
                    html.P("Nodes Executed", className="card-text text-muted")
                ])
            ], color="success", outline=True)
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{node_stats.get('total_duration', 0):.2f}s", className="card-title text-info"),
                    html.P("Total Duration", className="card-text text-muted")
                ])
            ], color="info", outline=True)
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{stats.get('avg_node_duration', 0):.3f}s", className="card-title text-warning"),
                    html.P("Avg Node Duration", className="card-text text-muted")
                ])
            ], color="warning", outline=True)
        ], width=3)
    ]

    return dbc.Row(cards, className="mb-4")


def create_timeline_figure(events_df: pd.DataFrame, node_durations: Dict[str, float]) -> go.Figure:
    """Create a clean, sequential execution flow visualization with completely separate plots."""
    if events_df.empty:
        return go.Figure()

    try:
        # Filter out step events (Graph step completed messages) - SAFER FILTERING
        # First check if 'message' column exists, if not, just filter by type
        if 'message' in events_df.columns:
            events_df = events_df[
                (events_df['type'] != 'step') &
                (~events_df['message'].str.contains('Graph step completed', na=False))
            ].copy()
        else:
            # If no message column, just filter by type
            events_df = events_df[events_df['type'] != 'step'].copy()

        # Create the main sequential execution flow figure
        fig = go.Figure()

        # === SEQUENTIAL EXECUTION FLOW - Main 2D plot ===
        # Sort events by timestamp for chronological order
        events_df_sorted = events_df.sort_values('start').reset_index(drop=True)


        # Create different y-levels for different node types
        unique_nodes = events_df_sorted[events_df_sorted['node'].notna()]['node'].unique()
        node_y_levels = {node: i + 1 for i, node in enumerate(unique_nodes)}

        # Add special y-levels for non-node events
        node_y_levels['pipeline'] = 0
        node_y_levels['system'] = len(unique_nodes) + 1

        # Group events by type for better visualization
        event_types = events_df_sorted['type'].unique()

        for event_type in event_types:
            type_df = events_df_sorted[events_df_sorted['type'] == event_type]
            if type_df.empty:
                continue

            color = get_event_color(event_type)

            # Clean marker styling
            marker_config = {
                'node_start': {'symbol': 'triangle-right', 'size': 16, 'line_width': 2},
                'node_end': {'symbol': 'triangle-left', 'size': 16, 'line_width': 2},
                'graph_event': {'symbol': 'diamond', 'size': 18, 'line_width': 2},
                'system': {'symbol': 'circle', 'size': 12, 'line_width': 1}
            }

            config = marker_config.get(event_type, {'symbol': 'circle', 'size': 14, 'line_width': 1})

            # Get sequential positions and y-levels for this event type
            type_indices = type_df.index.tolist()
            x_vals = type_indices

            # Assign y-levels based on node or event type
            y_vals = []
            for _, event in type_df.iterrows():
                node = event.get('node')
                if node and node in node_y_levels:
                    y_vals.append(node_y_levels[node])
                elif event_type == 'graph_event':
                    y_vals.append(node_y_levels['pipeline'])
                else:
                    y_vals.append(node_y_levels['system'])

            # Show legend for main event types
            show_legend = event_type in ['node_start', 'node_end', 'graph_event']

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(
                    color=color,
                    size=config['size'],
                    symbol=config['symbol'],
                    line=dict(width=config['line_width'], color='white'),
                    opacity=0.9
                ),
                name=f"{event_type.replace('_', ' ').title()}",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Position: %{x}<br>"
                    "Time: %{customdata[1]|%H:%M:%S.%3f}<br>"
                    "Type: " + event_type.replace('_', ' ').title() + "<br>"
                    "Details: %{customdata[2]}<br>"
                    "Log Index: %{customdata[3]}<br>"
                    "<extra></extra>"
                ),
                customdata=list(zip(type_df['event'], type_df['start'], type_df['details'], type_df['log_index'])),
                showlegend=show_legend
            ))

        # Add connecting lines to show execution flow (horizontal lines for each y-level)
        for y_level in set(node_y_levels.values()):
            events_at_level = []
            for i, event in events_df_sorted.iterrows():
                node = event.get('node')
                if node and node in node_y_levels and node_y_levels[node] == y_level:
                    events_at_level.append(i)
                elif event.get('type') == 'graph_event' and y_level == node_y_levels['pipeline']:
                    events_at_level.append(i)
                elif event.get('type') == 'system' and y_level == node_y_levels['system']:
                    events_at_level.append(i)

            if len(events_at_level) > 1:
                fig.add_trace(go.Scatter(
                    x=events_at_level,
                    y=[y_level] * len(events_at_level),
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0.2)', width=1, dash='dot'),
                    name=f'Flow Level {y_level}',
                    showlegend=False,
                    hoverinfo='skip'
                ))

        # === LAYOUT CONFIGURATION ===
        fig.update_layout(
            title={
                'text': "Sequential Execution Flow (2D)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'family': FONTS['primary'], 'color': COLORS['text_primary']},
                'pad': {'t': 30, 'b': 30}
            },
            height=600,
            hovermode='closest',
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation='v',
                yanchor='top',
                y=0.98,
                xanchor='left',
                x=1.02,
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1,
                font=dict(size=12, family=FONTS['primary']),
                title=dict(text="Event Types", font=dict(size=14, family=FONTS['primary']))
            ),
            font=dict(family=FONTS['primary'], size=12),
            paper_bgcolor='rgba(248,250,252,1)',
            plot_bgcolor='rgba(255,255,255,1)',
            margin=dict(l=120, r=200, t=150, b=100)
        )

        # === AXIS STYLING ===

        # X-axis: Execution sequence
        fig.update_xaxes(
            title_text="Execution Sequence (Step Order)",
            title_font=dict(size=16, family=FONTS['primary'], color=COLORS['text_primary']),
            tickmode='linear',
            tick0=0,
            dtick=1,  # Show every step
            gridcolor='rgba(0,0,0,0.1)',
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='rgba(0,0,0,0.2)',
            tickfont=dict(size=12, family=FONTS['mono']),
            showticklabels=False
        )

        # Y-axis: Execution levels
        y_labels = ['Pipeline Events']
        y_labels.extend([f"{node.title()}" for node in unique_nodes])
        y_labels.append('System Events')

        fig.update_yaxes(
            title_text="Execution Levels",
            title_font=dict(size=16, family=FONTS['primary']),
            tickvals=list(range(len(y_labels))),
            ticktext=y_labels,
            tickfont=dict(size=12, family=FONTS['primary'], color=COLORS['text_primary']),
            gridcolor='rgba(0,0,0,0.08)',
            showgrid=True,
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='rgba(0,0,0,0.2)',
            range=[-0.5, len(y_labels) - 0.5]
        )

        # Add summary annotation
        if not events_df.empty:
            total_duration = (pd.to_datetime(events_df['start'].max()) - 
                             pd.to_datetime(events_df['start'].min())).total_seconds()

            fig.add_annotation(
                text=(f"Pipeline Summary<br>Total Steps: {len(events_df)} | "
                      f"Duration: {total_duration:.2f}s | Nodes: {len(node_durations)}"),
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=14, color=COLORS['text_secondary'], family=FONTS['primary']),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                align="left"
            )

        return fig

    except Exception as e:
        # Create a more expressive error figure
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error Creating Timeline<br><br>"
                 f"<b>Error Details:</b><br>"
                 f"• {str(e)}<br><br>"
                 f"<b>DataFrame Info:</b><br>"
                 f"• Columns: {list(events_df.columns) if not events_df.empty else 'Empty DataFrame'}<br>"
                 f"• Shape: {events_df.shape}<br>"
                 f"• Types: {events_df.dtypes.to_dict() if not events_df.empty else 'No data'}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color=COLORS['danger'], family=FONTS['primary']),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=COLORS['danger'],
            borderwidth=2,
            align="center"
        )
        error_fig.update_layout(
            title="Timeline Creation Failed",
            height=400,
            showlegend=False,
            paper_bgcolor='rgba(248,250,252,1)',
            plot_bgcolor='rgba(255,255,255,1)'
        )
        return error_fig


def create_node_lifecycle_figure(events_df: pd.DataFrame) -> go.Figure:
    """Create a separate node lifecycle figure."""
    if events_df.empty:
        return go.Figure()

    try:
        # Filter out step events - SAFER FILTERING
        if 'message' in events_df.columns:
            events_df = events_df[
                (events_df['type'] != 'step') &
                (~events_df['message'].str.contains('Graph step completed', na=False))
            ].copy()
        else:
            # If no message column, just filter by type
            events_df = events_df[events_df['type'] != 'step'].copy()

        fig = go.Figure()

        # Show start-end pairs as connected lines
        nodes = events_df[events_df['type'].isin(['node_start', 'node_end'])]['node'].unique()
        nodes = [n for n in nodes if n is not None]

        # Sort events by timestamp for chronological order
        events_df_sorted = events_df.sort_values('start').reset_index(drop=True)

        for i, node in enumerate(nodes):
            node_events = events_df[
                (events_df['node'] == node) &
                (events_df['type'].isin(['node_start', 'node_end']))
            ].sort_values('start')

            if len(node_events) >= 2:
                start_event = node_events.iloc[0]
                end_event = node_events.iloc[-1]
                duration = (pd.to_datetime(end_event['start']) - pd.to_datetime(start_event['start'])).total_seconds()

                # Find sequential positions for start and end
                start_pos = events_df_sorted[events_df_sorted['log_index'] == start_event['log_index']].index[0]
                end_pos = events_df_sorted[events_df_sorted['log_index'] == end_event['log_index']].index[0]

                # Draw clean connection line
                fig.add_trace(go.Scatter(
                    x=[start_pos, end_pos],
                    y=[i, i],
                    mode='lines+markers+text',
                    line=dict(color=get_node_color(node), width=5, dash='solid'),
                    marker=dict(
                        color=[get_node_color(node), get_node_color(node)],
                        size=[16, 16],
                        symbol=['triangle-right', 'triangle-left'],
                        line=dict(width=2, color='white')
                    ),
                    text=['START', 'END'],
                    textposition=['middle right', 'middle left'],
                    textfont=dict(size=11, color=get_node_color(node), family=FONTS['mono']),
                    name=f"Connection {node.title()} Lifecycle",
                    hovertemplate=(
                        f"<b>{node.title()} Execution</b><br>"
                        f"Start Position: {start_pos}<br>"
                        f"End Position: {end_pos}<br>"
                        f"Duration: {duration:.3f}s<br>"
                        "<extra></extra>"
                    )
                ))

        # Layout for lifecycle figure
        fig.update_layout(
            title={
                'text': "Node Lifecycle View",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': FONTS['primary'], 'color': COLORS['text_primary']},
                'pad': {'t': 20, 'b': 20}
            },
            height=300,
            hovermode='closest',
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation='v',
                yanchor='top',
                y=0.98,
                xanchor='left',
                x=1.02,
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1,
                font=dict(size=11, family=FONTS['primary'])
            ),
            font=dict(family=FONTS['primary'], size=12),
            paper_bgcolor='rgba(248,250,252,1)',
            plot_bgcolor='rgba(255,255,255,1)',
            margin=dict(l=120, r=200, t=100, b=80)
        )

        # X-axis: Execution sequence
        fig.update_xaxes(
            title_text="Node Execution Sequence",
            title_font=dict(size=14, family=FONTS['primary']),
            tickmode='linear',
            tick0=0,
            dtick=1,
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            tickfont=dict(size=11, family=FONTS['mono']),
            showticklabels=False
        )

        # Y-axis: Node connections
        fig.update_yaxes(
            title_text="Node Connections",
            title_font=dict(size=14, family=FONTS['primary']),
            tickvals=list(range(len(nodes))) if nodes else [],
            ticktext=[f"Node {node.title()}" for node in nodes] if nodes else [],
            tickfont=dict(size=11, family=FONTS['primary']),
            gridcolor='rgba(0,0,0,0.05)',
            showgrid=True,
            showline=True,
            linewidth=2,
            linecolor='rgba(0,0,0,0.2)'
        )

        return fig

    except Exception as e:
        # Create a more expressive error figure
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error Creating Lifecycle View<br><br>"
                 f"<b>Error Details:</b><br>"
                 f"• {str(e)}<br><br>"
                 f"<b>DataFrame Info:</b><br>"
                 f"• Columns: {list(events_df.columns) if not events_df.empty else 'Empty DataFrame'}<br>"
                 f"• Shape: {events_df.shape}<br>"
                 f"• Types: {events_df.dtypes.to_dict() if not events_df.empty else 'No data'}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color=COLORS['danger'], family=FONTS['primary']),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=COLORS['danger'],
            borderwidth=2,
            align="center"
        )
        error_fig.update_layout(
            title="Lifecycle Creation Failed",
            height=300,
            showlegend=False,
            paper_bgcolor='rgba(248,250,252,1)',
            plot_bgcolor='rgba(255,255,255,1)'
        )
        return error_fig


def create_performance_figure(node_durations: Dict[str, float]) -> go.Figure:
    """Create a separate performance figure."""
    if not node_durations:
        return go.Figure()

    fig = go.Figure()

    nodes = list(node_durations.keys())
    durations = list(node_durations.values())
    colors = [get_node_color(node) for node in nodes]

    # Sort by duration for better visualization
    sorted_data = sorted(zip(nodes, durations, colors), key=lambda x: x[1], reverse=True)
    nodes, durations, colors = zip(*sorted_data)

    # Create clean performance labels
    performance_labels = [f"Node {node.title()}" for node in nodes]

    fig.add_trace(go.Bar(
        x=durations,
        y=performance_labels,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='white', width=1),
            opacity=0.8
        ),
        text=[f"{d:.3f}s" for d in durations],
        textposition='auto',
        textfont=dict(color='white', size=12, family=FONTS['mono']),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Duration: %{x:.3f} seconds<br>"
            "<extra></extra>"
        )
    ))

    # Layout for performance figure
    fig.update_layout(
        title={
            'text': "Performance Metrics",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': FONTS['primary'], 'color': COLORS['text_primary']},
            'pad': {'t': 20, 'b': 20}
        },
        height=300,
        template='plotly_white',
        showlegend=False,
        font=dict(family=FONTS['primary'], size=12),
        paper_bgcolor='rgba(248,250,252,1)',
        plot_bgcolor='rgba(255,255,255,1)',
        margin=dict(l=120, r=100, t=100, b=80)
    )

    # X-axis: Duration
    fig.update_xaxes(
        title_text="Execution Duration (seconds)",
        title_font=dict(size=14, family=FONTS['primary']),
        gridcolor='rgba(0,0,0,0.1)',
        showgrid=True,
        tickformat='.3f',
        tickfont=dict(size=11, family=FONTS['mono']),
        showline=True,
        linewidth=2,
        linecolor='rgba(0,0,0,0.2)'
    )

    # Y-axis: Performance ranking
    fig.update_yaxes(
        title_text="Performance Ranking",
        title_font=dict(size=14, family=FONTS['primary']),
        tickfont=dict(size=11, family=FONTS['primary']),
        gridcolor='rgba(0,0,0,0.05)',
        showgrid=True,
        showline=True,
        linewidth=2,
        linecolor='rgba(0,0,0,0.2)'
    )

    return fig


def create_json_viewer(data: Any, depth: int = 0, max_depth: int = 8) -> html.Div:
    """Create an enhanced JSON viewer with syntax highlighting."""
    if depth > max_depth:
        return html.Span("... (truncated)", style={'color': COLORS['text_secondary'], 'fontStyle': 'italic'})

    if isinstance(data, dict):
        if not data:
            return html.Span("{}", style={'color': COLORS['text_secondary']})

        children = []
        for i, (key, value) in enumerate(data.items()):
            children.append(html.Div([
                html.Span(f'"{key}"', style={
                    'color': COLORS['info'],
                    'fontWeight': '600'
                }),
                html.Span(': ', style={'color': COLORS['text_secondary']}),
                create_json_viewer(value, depth + 1, max_depth)
            ], style={'marginLeft': f'{depth * 20}px' if depth > 0 else '0px'}))

        return html.Div(children, style={
            'borderLeft': f'2px solid {COLORS["border"]}' if depth > 0 else 'none',
            'paddingLeft': '12px' if depth > 0 else '0px',
            'marginTop': '4px' if depth > 0 else '0px'
        })

    elif isinstance(data, list):
        if not data:
            return html.Span("[]", style={'color': COLORS['text_secondary']})

        children = []
        for i, item in enumerate(data):
            children.append(html.Div([
                html.Span(f'[{i}] ', style={'color': COLORS['warning'], 'fontWeight': '500'}),
                create_json_viewer(item, depth + 1, max_depth)
            ], style={'marginLeft': f'{depth * 20}px' if depth > 0 else '0px'}))

        return html.Div(children, style={
            'borderLeft': f'2px solid {COLORS["border"]}' if depth > 0 else 'none',
            'paddingLeft': '12px' if depth > 0 else '0px',
            'marginTop': '4px' if depth > 0 else '0px'
        })

    else:
        # Primitive values
        color = COLORS['text_primary']  # Default
        if isinstance(data, str):
            color = COLORS['success']
            data = f'"{data}"'
        elif isinstance(data, bool):
            color = COLORS['warning']
            data = str(data).lower()
        elif isinstance(data, (int, float)):
            color = COLORS['info']
            data = str(data)
        elif data is None:
            color = COLORS['text_secondary']
            data = 'null'

        return html.Span(
            str(data),
            style={
                'color': color,
                'fontWeight': '500' if isinstance(data, bool) else 'normal'
            }
        )


def create_log_detail_tabs(log_entry: Dict[str, Any]) -> dbc.Card:
    """Create enhanced tabbed interface for log details."""
    state_data = log_entry.get('data', {}).get('state', {})
    result_data = log_entry.get('data', {}).get('result', {})
    message_data = log_entry.get('data', {})

    tabs = [
        dbc.Tab(
            label="State Data",
            tab_id="state-tab",
            children=[
                html.Div([
                    html.H5("Current State", className="mb-3"),
                    html.Div(
                        create_json_viewer(state_data),
                        style=JSON_VIEWER_STYLE
                    ) if state_data else html.P("No state data available", className="text-muted")
                ], style={'padding': '20px'})
            ]
        ),

        dbc.Tab(
            label="Results",
            tab_id="result-tab",
            children=[
                html.Div([
                    html.H5("Node Results", className="mb-3"),
                    html.Div(
                        create_json_viewer(result_data),
                        style=JSON_VIEWER_STYLE
                    ) if result_data else html.P("No result data available", className="text-muted")
                ], style={'padding': '20px'})
            ]
        ),

        dbc.Tab(
            label="Message",
            tab_id="message-tab",
            children=[
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Timestamp:", className="text-muted"),
                            html.P(log_entry.get('timestamp', 'N/A'), className="font-monospace")
                        ], width=6),
                        dbc.Col([
                            html.H6("Message:", className="text-muted"),
                            html.P(log_entry.get('message', 'N/A'), className="fw-bold")
                        ], width=6)
                    ], className="mb-3"),

                    html.H5("Full Message Data", className="mb-3"),
                    html.Div(
                        create_json_viewer(message_data),
                        style=JSON_VIEWER_STYLE
                    )
                ], style={'padding': '20px'})
            ]
        ),

        dbc.Tab(
            label="Raw Log",
            tab_id="raw-tab",
            children=[
                html.Div([
                    html.H5("Complete Log Entry", className="mb-3"),
                    html.Div(
                        create_json_viewer(log_entry),
                        style=JSON_VIEWER_STYLE
                    )
                ], style={'padding': '20px'})
            ]
        )
    ]

    return dbc.Card([
        dbc.CardHeader([
            html.H4("Log Entry Details", className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Tabs(tabs, id="detail-tabs", active_tab="state-tab")
        ])
    ])


def create_empty_state() -> html.Div:
    """Create empty state component with placeholder timeline."""
    return html.Div([
        # Timeline card with placeholder
        dbc.Card([
            dbc.CardHeader([
                html.H4("Execution Timeline", className="mb-0")
            ]),
            dbc.CardBody([
                # Empty placeholder graph to prevent callback errors
                dcc.Graph(
                    id='timeline-graph',
                    figure=go.Figure().add_annotation(
                        text="Upload a log file to view the execution timeline",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=18, color=COLORS['text_secondary'])
                    ).update_layout(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=1200  # Match the new timeline height
                    ),
                    config={'displayModeBar': False},
                    style={'height': '1200px'}
                )
            ])
        ], className="mb-4"),

        # Instruction card
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-cloud-upload-alt fa-3x text-muted mb-3"),
                    html.H4("No Data to Display", className="text-muted"),
                    html.P("Upload a log file to start analyzing your pipeline execution.", className="text-muted")
                ], style={'textAlign': 'center', 'padding': '40px'})
            ])
        ], className="bg-light")
    ])


def create_error_alert(message: str) -> dbc.Alert:
    """Create error alert component."""
    return dbc.Alert([
        html.I(className="fas fa-exclamation-triangle me-2"),
        html.Strong("Error: "),
        message
    ], color="danger", dismissable=True)
