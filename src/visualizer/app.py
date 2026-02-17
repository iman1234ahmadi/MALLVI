"""Main Dash application for the modernized log visualizer."""

import base64
from typing import Optional, Dict, Any
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, no_update, callback_context, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from .data_processor import LogProcessor
from .components import (
    create_timeline_figure,
    create_node_lifecycle_figure,
    create_performance_figure,
    create_header,
    create_upload_component,
    create_empty_state,
    create_error_alert,
    create_stats_cards,
    create_log_detail_tabs
)
from .styles import (
    COLORS, LAYOUT_STYLE, TIMELINE_CONFIG
)


def create_app(debug: bool = True) -> Dash:
    """Create and configure the Dash application."""

    # Initialize with Bootstrap theme
    app = Dash(
        __name__,
        title='Robotics Pipeline Analyzer',
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
            'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap',
            'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap'
        ],
        suppress_callback_exceptions=True,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"},
            {"name": "description", "content": "Interactive robotics pipeline log analyzer"}
        ]
    )

    # Custom CSS for enhanced styling
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                    background-color: #f8fafc;
                }
                .dash-loading {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 200px;
                    flex-direction: column;
                }
                .fa-spin {
                    animation: fa-spin 2s infinite linear;
                }
                @keyframes fa-spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                .tab-content {
                    border: 1px solid #e2e8f0;
                    border-top: none;
                    border-radius: 0 0 8px 8px;
                }
                .nav-tabs .nav-link.active {
                    background-color: #1e40af !important;
                    border-color: #1e40af !important;
                    color: white !important;
                }
                .nav-tabs .nav-link {
                    border-radius: 8px 8px 0 0 !important;
                    border: 1px solid #e2e8f0 !important;
                    margin-right: 4px;
                }
                .card {
                    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06) !important;
                    border: 1px solid #e2e8f0 !important;
                }
                .upload-component:hover {
                    border-color: #1e40af !important;
                    background-color: #f1f5f9 !important;
                }
                /* Custom loading spinner */
                .loading-spinner {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 40px;
                }
                .loading-spinner .fa-cog {
                    animation: spin-slow 3s infinite linear;
                    color: #1e40af;
                }
                @keyframes spin-slow {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''

    # Create layout
    app.layout = create_layout()

    # Register callbacks
    register_callbacks(app)

    return app


def create_layout() -> dbc.Container:
    """Create the main application layout with sidebar modal."""
    return dbc.Container([
        # Stores for data
        dcc.Store(id='log-store'),
        dcc.Store(id='events-store'),
        dcc.Store(id='stats-store'),

        # Sidebar modal for log details
        dbc.Modal([
            dbc.ModalHeader([
                dbc.ModalTitle("Log Entry Details", id="modal-title")
            ]),
            dbc.ModalBody([
                html.Div(id='modal-log-details')
            ], style={'maxHeight': '70vh', 'overflowY': 'auto'}),
            dbc.ModalFooter([
                dbc.Button("Close", id="close-modal", className="ms-auto", n_clicks=0)
            ])
        ], id="log-details-modal", size="xl", is_open=False),

        # Header
        create_header(),

        # Upload section
        create_upload_component(),

        # File info section
        html.Div(id='file-info-section'),

        # Stats cards
        html.Div(id='stats-cards-section'),

        # Main content area with custom loading spinner
        dcc.Loading(
            id="loading-main",
            type="default",
            color=COLORS['primary'],
            style={'minHeight': '200px'},
            children=[
                html.Div(id='main-content', children=[create_empty_state()])
            ],
            custom_spinner=html.Div([
                html.Div([
                    html.I(className="fas fa-cog fa-spin fa-3x", style={'color': COLORS['primary']}),
                    html.P("Analyzing pipeline logs...", className="mt-3", style={'color': COLORS['text_secondary']})
                ], style={'textAlign': 'center', 'padding': '60px'})
            ])
        ),

        # Footer
        html.Hr(style={'marginTop': '48px'}),
        html.Footer([
            dbc.Row([
                dbc.Col([
                    html.P([
                        "Robotics Pipeline Analyzer v2.0 | ",
                        html.A("Built with Dash", href="https://dash.plotly.com/", target="_blank"),
                        " & ",
                        html.A("Bootstrap", href="https://getbootstrap.com/", target="_blank")
                    ], className="text-muted text-center mb-0")
                ])
            ])
        ], style={'padding': '24px 0'})

    ], fluid=True, style=LAYOUT_STYLE)


def _create_empty_modal_content() -> html.Div:
    """Create empty state content for modal."""
    return html.Div([
        html.Div([
            html.I(className="fas fa-mouse-pointer fa-3x text-muted mb-3"),
            html.H5("Select an Event", className="text-muted"),
            html.P("Click on any event point in the timeline to view detailed log information.",
                  className="text-muted")
        ], style={'textAlign': 'center', 'padding': '60px'})
    ])


def register_callbacks(app: Dash) -> None:
    """Register all application callbacks."""

    @app.callback(
        [Output('log-store', 'data'),
         Output('file-info-section', 'children')],
        [Input('upload-data', 'contents')],
        [State('upload-data', 'filename')]
    )
    def upload_log_file(contents: Optional[str], filename: Optional[str]):
        """Handle log file upload."""
        if not contents or not filename:
            raise PreventUpdate

        try:
            # Decode file contents
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string).decode('utf-8')
            log_lines = decoded.splitlines()

            # Process logs
            processor = LogProcessor()
            logs = processor.parse_logs(log_lines)

            if not logs:
                return no_update, create_error_alert("No valid log entries found in the uploaded file.")

            # Create file info card
            file_info = dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("File Information", className="card-title"),
                            html.P(f"Filename: {filename}", className="mb-1"),
                            html.P(f"Log entries: {len(logs):,}", className="mb-1"),
                            html.P(f"Size: {len(content_string)} bytes", className="mb-0")
                        ], width=6),
                        dbc.Col([
                            html.H6("Timeline:", className="text-muted mb-2"),
                            html.P(f"Start: {logs[0]['timestamp']}", className="mb-1 font-monospace small"),
                            html.P(f"End: {logs[-1]['timestamp']}", className="mb-0 font-monospace small")
                        ], width=6)
                    ])
                ])
            ], className="mb-4", color="success", outline=True)

            return {'logs': logs, 'filename': filename}, file_info

        except Exception as e:
            error_msg = f"Failed to process log file: {str(e)}"
            return no_update, create_error_alert(error_msg)

    @app.callback(
        [Output('events-store', 'data'),
         Output('stats-store', 'data')],
        [Input('log-store', 'data')]
    )
    def process_log_data(log_data: Optional[Dict[str, Any]]):
        """Process logs into events and statistics."""
        if not log_data:
            raise PreventUpdate

        try:
            processor = LogProcessor()
            processor.logs = log_data['logs']

            # Create events dataframe
            events_df, node_durations = processor.create_events_dataframe()

            # Get statistics
            stats = processor.get_summary_stats()
            performance = processor.get_node_performance()

            events_data = {
                'events_df': events_df.to_dict('records'),
                'node_durations': node_durations
            }

            stats_data = {
                'summary': stats,
                'performance': performance
            }

            return events_data, stats_data

        except Exception as e:
            print(f"Error processing log data: {e}")
            raise PreventUpdate

    @app.callback(
        Output('stats-cards-section', 'children'),
        [Input('stats-store', 'data')]
    )
    def update_stats_cards(stats_data: Optional[Dict[str, Any]]):
        """Update statistics cards."""
        if not stats_data:
            return html.Div()

        stats = stats_data.get('summary', {})
        return create_stats_cards(stats)

    @app.callback(
        Output('main-content', 'children'),
        [Input('events-store', 'data')]
    )
    def update_main_content(events_data: Optional[Dict[str, Any]]):
        """Update main content area with timeline and details."""
        if not events_data:
            return create_empty_state()

        try:
            events_df = pd.DataFrame(events_data['events_df'])
            node_durations = events_data['node_durations']

            # Create timeline figure
            timeline_fig = create_timeline_figure(events_df, node_durations)

            return html.Div([
                # Main sequential execution flow
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Sequential Execution Flow", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id={'type': 'graph', 'index': 'timeline'},
                            figure=timeline_fig,
                            config=TIMELINE_CONFIG,
                            style={'height': '600px'}
                        )
                    ])
                ], className="mb-4"),

                # Node lifecycle view
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Node Lifecycle View", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id={'type': 'graph', 'index': 'lifecycle'},
                            figure=create_node_lifecycle_figure(events_df),
                            config=TIMELINE_CONFIG,
                            style={'height': '300px'}
                        )
                    ])
                ], className="mb-4"),

                # Performance metrics
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Performance Metrics", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(
                            id={'type': 'graph', 'index': 'performance'},
                            figure=create_performance_figure(node_durations),
                            config=TIMELINE_CONFIG,
                            style={'height': '300px'}
                        )
                    ])
                ], className="mb-4"),

                # Small instruction card
                dbc.Card([
                    dbc.CardBody([
                        html.P([
                            html.I(className="fas fa-info-circle me-2"),
                            "Click on any event point in the timeline to view detailed log "
                            "information in the sidebar modal."
                        ], className="text-muted mb-0", style={'textAlign': 'center'})
                    ])
                ], className="bg-light")
            ])

        except Exception as e:
            return create_error_alert(f"Failed to create timeline: {str(e)}")

    @app.callback(
        [Output('log-details-modal', 'is_open'),
         Output('modal-log-details', 'children'),
         Output('modal-title', 'children')],
        [Input({'type': 'graph', 'index': ALL}, 'clickData'),
         Input('close-modal', 'n_clicks')],
        [State('log-details-modal', 'is_open'),
         State('log-store', 'data')]
    )
    def handle_modal_interactions(graph_clicks, close_clicks, is_open, log_data):
        """Handle modal open/close and log details display."""
        ctx = callback_context
        if not ctx.triggered:
            return False, _create_empty_modal_content(), "Log Entry Details"

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Close modal
        if trigger_id == 'close-modal':
            return False, _create_empty_modal_content(), "Log Entry Details"

        # Handle graph clicks (any graph that was clicked)
        # Test pattern matching more carefully
        is_graph_click = False
        try:
            import json
            parsed_trigger = json.loads(trigger_id)
            is_graph_click = parsed_trigger.get('type') == 'graph'
        except (json.JSONDecodeError, TypeError):
            is_graph_click = False

        if is_graph_click and log_data:
            # Find which graph was clicked
            for i, click_data in enumerate(graph_clicks):
                if click_data:
                    try:
                        point = click_data['points'][0]
                        customdata = point.get('customdata')

                        # Handle different click types
                        if customdata is not None:
                            # Check if customdata is a list (new format) or single value (old format)
                            if isinstance(customdata, list) and len(customdata) >= 4:
                                # New format: [node_type, timestamp, message, log_index]
                                log_index = customdata[3]
                            elif isinstance(customdata, list) and len(customdata) >= 3:
                                # Fallback format: [node_type, timestamp, log_index] or similar
                                # Try to find the numeric value
                                for item in customdata:
                                    if isinstance(item, (int, float)):
                                        log_index = item
                                        break
                                else:
                                    log_index = None
                            elif isinstance(customdata, (int, float)):
                                log_index = customdata
                            else:
                                log_index = None

                            if log_index is not None and isinstance(log_index, (int, float)):
                                # Event marker clicked
                                log_entry = log_data['logs'][int(log_index)]
                                modal_content = create_log_detail_tabs(log_entry)
                                return True, modal_content, f"Log Entry Details - {log_entry.get('message', 'Unknown')}"

                        elif 'y' in point and 'x' in point:
                            # Duration bar clicked (from performance graph)
                            node_name = str(point['y'])
                            try:
                                duration = float(point['x'])
                            except (ValueError, TypeError):
                                duration_str = str(point.get('text', '0'))
                                if 'seconds' in duration_str:
                                    import re
                                    match = re.search(r'(\d+\.?\d*)\s*seconds?', duration_str)
                                    duration = float(match.group(1)) if match else 0.0
                                else:
                                    duration = 0.0

                            performance_content = dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.H6("Node Name:", className="text-muted"),
                                            html.P(node_name, className="fw-bold")
                                        ], width=6),
                                        dbc.Col([
                                            html.H6("Execution Time:", className="text-muted"),
                                            html.P(f"{duration:.3f} seconds", className="fw-bold text-primary")
                                        ], width=6)
                                    ]),
                                    html.Hr(),
                                    html.P("This shows the total execution time for this node in the pipeline.",
                                          className="text-muted mb-0")
                                ])
                            ])

                            return True, performance_content, f"{node_name.title()} Performance"

                    except Exception as e:
                        error_content = create_error_alert(f"Error displaying log details: {str(e)}")
                        return True, error_content, "Error"

        return is_open, _create_empty_modal_content(), "Log Entry Details"


def main():
    """Main entry point for running the application."""
    app = create_app(debug=True)
    print("Starting Robotics Pipeline Analyzer...")
    print("Navigate to http://localhost:8050 to view the application")
    app.run(debug=True, host='0.0.0.0', port=8050)


if __name__ == '__main__':
    main()
