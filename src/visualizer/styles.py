"""Modern styling constants and utilities for the log visualizer."""

# Modern color palette
COLORS = {
    # Node events
    'decomposer': '#6366f1',    # Indigo
    'perceptor': '#8b5cf6',     # Violet
    'grounder': '#06b6d4',      # Cyan
    'segmentor': '#10b981',     # Emerald
    'projector': '#f59e0b',     # Amber
    'thinker': '#ef4444',       # Red
    'actor': '#ec4899',         # Pink
    'reflector': '#84cc16',     # Lime

    # Event types
    'node_start': '#3b82f6',    # Blue
    'node_end': '#10b981',      # Green
    'graph_event': '#ef4444',   # Red
    'step': '#8b5cf6',          # Purple
    'system': '#6b7280',        # Gray

    # UI colors
    'primary': '#1e40af',       # Blue 800
    'secondary': '#6b7280',     # Gray 500
    'success': '#059669',       # Green 600
    'warning': '#d97706',       # Orange 600
    'danger': '#dc2626',        # Red 600
    'info': '#0284c7',          # Sky 600

    # Background colors
    'bg_primary': '#ffffff',
    'bg_secondary': '#f8fafc',
    'bg_tertiary': '#f1f5f9',
    'border': '#e2e8f0',
    'text_primary': '#1e293b',
    'text_secondary': '#64748b'
}

# Typography
FONTS = {
    'primary': "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif",
    'mono': "'JetBrains Mono', 'Fira Code', 'Monaco', 'Consolas', monospace"
}

# Component styles
CARD_STYLE = {
    'backgroundColor': COLORS['bg_primary'],
    'border': f"1px solid {COLORS['border']}",
    'borderRadius': '12px',
    'padding': '24px',
    'marginBottom': '24px',
    'boxShadow': '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)'
}

UPLOAD_STYLE = {
    'width': '100%',
    'height': '120px',
    'lineHeight': '120px',
    'borderWidth': '2px',
    'borderStyle': 'dashed',
    'borderRadius': '12px',
    'textAlign': 'center',
    'margin': '20px 0',
    'borderColor': COLORS['border'],
    'backgroundColor': COLORS['bg_secondary'],
    'cursor': 'pointer',
    'fontSize': '16px',
    'fontFamily': FONTS['primary'],
    'color': COLORS['text_secondary'],
    'transition': 'all 0.2s ease-in-out'
}

UPLOAD_STYLE_HOVER = {
    **UPLOAD_STYLE,
    'borderColor': COLORS['primary'],
    'backgroundColor': COLORS['bg_tertiary'],
    'color': COLORS['primary']
}

TAB_STYLE = {
    'borderRadius': '8px 8px 0 0',
    'padding': '12px 24px',
    'fontFamily': FONTS['primary'],
    'fontWeight': '500',
    'border': 'none'
}

TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    'backgroundColor': COLORS['primary'],
    'color': 'white'
}

JSON_VIEWER_STYLE = {
    'backgroundColor': COLORS['bg_tertiary'],
    'padding': '20px',
    'borderRadius': '8px',
    'maxHeight': '500px',
    'overflowY': 'auto',
    'fontFamily': FONTS['mono'],
    'fontSize': '14px',
    'lineHeight': '1.5',
    'border': f"1px solid {COLORS['border']}"
}

TIMELINE_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'responsive': True,
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'robotics_pipeline_timeline',
        'height': 1200,
        'width': 1600,
        'scale': 2
    },
    'scrollZoom': True,
    'doubleClick': 'reset+autosize'
}

LAYOUT_STYLE = {
    'maxWidth': '1400px',
    'margin': '0 auto',
    'padding': '32px 24px',
    'fontFamily': FONTS['primary'],
    'backgroundColor': COLORS['bg_secondary'],
    'minHeight': '100vh'
}

HEADER_STYLE = {
    'textAlign': 'center',
    'marginBottom': '32px',
    'color': COLORS['text_primary']
}

TITLE_STYLE = {
    'fontSize': '2.5rem',
    'fontWeight': '700',
    'marginBottom': '8px',
    'background': f"linear-gradient(135deg, {COLORS['primary']}, {COLORS['info']})",
    'WebkitBackgroundClip': 'text',
    'WebkitTextFillColor': 'transparent',
    'backgroundClip': 'text'
}

SUBTITLE_STYLE = {
    'fontSize': '1.125rem',
    'color': COLORS['text_secondary'],
    'fontWeight': '400'
}

def get_node_color(node_name):
    """Get color for a specific node."""
    return COLORS.get(node_name.lower(), COLORS['secondary'])

def get_event_color(event_type):
    """Get color for a specific event type."""
    return COLORS.get(event_type, COLORS['secondary'])
