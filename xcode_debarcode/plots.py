"""Interactive visualization functions for debarcoding analysis."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import anndata as ad
from typing import Optional, List
import warnings

__all__ = [
    "plot_gate_intensities", 
    "plot_confidence_distribution", 
    "plot_hamming_graph",
    "plot_barcode_rank_histogram",
    "plot_hamming_heatmap"
]


def plot_barcode_rank_histogram(adata: ad.AnnData,
                                assignment_col: str,
                                top_n: int = 20,
                                min_count: int = 1) -> go.Figure:
    """Plot rank histogram of top barcode patterns.
    
    Shows the most frequent barcode patterns as a ranked bar chart.
    Valid patterns shown in blue, invalid patterns in grey.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    assignment_col : str
        Column name with barcode assignments (string patterns)
    top_n : int
        Number of top patterns to show (default: 20)
    min_count : int
        Minimum count to include (default: 1)
    
    Returns:
    --------
    fig : plotly Figure
        Interactive bar chart
    """
    from collections import Counter
    from .barcode import is_valid_pattern
    
    if assignment_col not in adata.obs.columns:
        raise ValueError(f"Assignment column '{assignment_col}' not found in adata.obs")
    
    patterns_str = adata.obs[assignment_col].values
    patterns = [tuple(int(c) for c in p) for p in patterns_str]
    pattern_counts = Counter(patterns)
    
    filtered = {p: c for p, c in pattern_counts.items() if c >= min_count}
    top_patterns = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not top_patterns:
        raise ValueError(f"No patterns with count >= {min_count}")
    
    ranks = list(range(1, len(top_patterns) + 1))
    counts = [c for _, c in top_patterns]
    colors = ['#3498db' if is_valid_pattern(p) else '#95a5a6' for p, _ in top_patterns]
    hover_texts = [
        f"Rank: {r}<br>Pattern: {{{', '.join(str(i+1) for i, b in enumerate(p) if b)}}}<br>"
        f"Count: {c}<br>Valid: {is_valid_pattern(p)}"
        for r, (p, c) in enumerate(top_patterns, 1)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=ranks,
            y=counts,
            marker=dict(color=colors, line=dict(color='#2c3e50', width=1)),
            customdata=hover_texts,
            hovertemplate='%{customdata}<extra></extra>',
            showlegend=False
        )
    )
    
    fig.update_layout(
        title=f"Barcode Pattern Rank Histogram<br>"
              f"<sub>Top {len(top_patterns)} patterns | {sum(counts):,} cells | min_count={min_count}</sub>",
        xaxis=dict(
            title="Rank (by frequency)",
            tickmode='linear',
            tick0=1,
            dtick=1 if len(top_patterns) <= 20 else 5
        ),
        yaxis_title="Count",
        hovermode='closest',
        height=500
    )
    
    return fig

def plot_barcode_rank_histogram(adata: ad.AnnData,
                                assignment_col: str,
                                top_n: int = 20,
                                min_count: int = 1) -> go.Figure:
    """Plot rank histogram of top barcode patterns.
    
    Shows the most frequent barcode patterns as a ranked bar chart.
    Valid patterns shown in blue, invalid patterns in grey.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    assignment_col : str
        Column name with barcode assignments (string patterns)
    top_n : int
        Number of top patterns to show (default: 20)
    min_count : int
        Minimum count to include (default: 1)
    
    Returns:
    --------
    fig : plotly Figure
        Interactive bar chart
    """
    from collections import Counter
    from .barcode import is_valid_pattern
    
    # Validate column
    if assignment_col not in adata.obs.columns:
        raise ValueError(f"Assignment column '{assignment_col}' not found in adata.obs")
    
    # Get patterns and counts
    patterns_str = adata.obs[assignment_col].values
    patterns = [tuple(int(c) for c in p) for p in patterns_str]
    pattern_counts = Counter(patterns)
    
    # Filter by min_count and get top N
    filtered = {p: c for p, c in pattern_counts.items() if c >= min_count}
    top_patterns = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if len(top_patterns) == 0:
        raise ValueError(f"No patterns with count >= {min_count}")
    
    # Prepare data
    ranks = list(range(1, len(top_patterns) + 1))
    counts = []
    hover_texts = []
    colors = []
    
    for rank, (pattern, count) in enumerate(top_patterns, 1):
        counts.append(count)
        
        # Pattern as set
        pattern_indices = [i+1 for i, bit in enumerate(pattern) if bit == 1]
        
        # Check validity
        is_valid = is_valid_pattern(pattern)
        
        # Color: blue for valid, grey for invalid
        colors.append('#3498db' if is_valid else '#95a5a6')
        
        # Hover text
        hover_text = (
            f"Rank: {rank}<br>"
            f"Pattern: {set(pattern_indices)}<br>"
            f"Count: {count}<br>"
            f"Valid: {is_valid}"
        )
        hover_texts.append(hover_text)
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=ranks,
            y=counts,
            marker=dict(
                color=colors,
                line=dict(color='#2c3e50', width=1)
            ),
            customdata=hover_texts,
            hovertemplate='%{customdata}<extra></extra>',
            showlegend=False
        )
    )
    
    # Update layout
    total_cells = sum(counts)
    
    fig.update_layout(
        title=dict(
            text=f"Barcode Pattern Rank Histogram<br>"
                 f"<sub>Top {len(top_patterns)} patterns | {total_cells:,} cells | min_count={min_count}</sub>",
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Rank (by frequency)",
            tickmode='linear',
            tick0=1,
            dtick=1 if len(top_patterns) <= 20 else 5
        ),
        yaxis=dict(
            title="Count"
        ),
        hovermode='closest',
        height=500,
        showlegend=False
    )
    
    return fig


def plot_gate_intensities(adata: ad.AnnData,
                         gates: Optional[List[str]] = None,
                         layer: Optional[str] = None,
                         show_method_data: Optional[str] = None,
                         log_scale_x: bool = True,
                         log_scale_y: bool = False,
                         bins: int = 250,
                         xlim: Optional[tuple] = None,
                         xlim_percentile: tuple = (0.1, 99.9)) -> go.Figure:
    """Plot interactive gate intensity distributions.
    
    Creates interactive histograms for barcode channel intensities.
    Can overlay learned parameters from debarcoding methods.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    gates : list of str, optional
        Specific gates to visualize. If None, shows all barcode channels.
    layer : str, optional
        Layer to use for visualization. If None, uses .X (raw data).
        If using a transformed layer (e.g., 'log', 'arcsinh_cf5'), 
        set log_scale_x=False since data is already transformed.
    show_method_data : str, optional
        Name of debarcoding result from adata.uns['debarcoding'] to visualize.
        This is the method_name used when calling debarcode().
        
        Shows overlays based on the method:
        - GMM: OFF/ON means (red/green) + threshold (yellow)
        - PC-GMM: OFF/ON means (red/green)
        - Manual: threshold (yellow)
        - Auto -> PC-GMM: OFF/ON means (red/green)
        - Auto -> Scoring: no overlays
        - PreMessa/Scoring: no overlays
    log_scale_x : bool
        Use log scale for x-axis (default: True).
        Set to False when using transformed layers.
    log_scale_y : bool
        Use log scale for y-axis (default: False)
    bins : int
        Number of histogram bins (default: 250)
    xlim : tuple, optional
        X-axis limits as (xmin, xmax) in the data space (layer or raw).
        Supports partial specification:
        - None: automatic (uses xlim_percentile)
        - (xmin, xmax): both limits fixed
        - (None, xmax): automatic min, fixed max
        - (xmin, None): fixed min, automatic max
    xlim_percentile : tuple
        Percentile range for automatic limits (default: (0.1, 99.9))
    
    Returns:
    --------
    fig : Interactive plotly figure
    """
    
    from .io import get_barcode_channels
    barcode_channels = get_barcode_channels(adata)
    
    if gates is None:
        gates_to_plot = barcode_channels
    else:
        invalid = set(gates) - set(barcode_channels)
        if invalid:
            raise ValueError(f"Invalid gates (not barcode channels): {invalid}")
        gates_to_plot = gates
    
    n_gates = len(gates_to_plot)
    n_cols = min(6, n_gates)
    n_rows = int(np.ceil(n_gates / n_cols))
    
    is_transformed = (layer is not None and layer in adata.layers and 
                     (layer.startswith('log') or layer.startswith('arcsinh')))
    
    if is_transformed and log_scale_x:
        warnings.warn(
            f"Using transformed layer '{layer}' with log_scale_x=True may produce "
            f"unexpected results. Consider using log_scale_x=False.",
            UserWarning
        )
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=gates_to_plot,
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    method_type = None
    gate_params_dict = None
    prob_thresh = 0.5
    
    if show_method_data:
        if 'debarcoding' not in adata.uns:
            warnings.warn(
                "No debarcoding metadata found in adata.uns['debarcoding']. "
                "No method overlays will be shown.",
                UserWarning
            )
        elif show_method_data not in adata.uns['debarcoding']:
            available = list(adata.uns['debarcoding'].keys())
            warnings.warn(
                f"Method '{show_method_data}' not found in adata.uns['debarcoding']. "
                f"Available methods: {available}. No method overlays will be shown.",
                UserWarning
            )
        else:
            method_info = adata.uns['debarcoding'][show_method_data]
            base_method = method_info.get('method')
            
            if base_method == 'gmm':
                method_type = 'gmm'
                gate_params_dict = method_info.get('gate_params')
                prob_thresh = method_info.get('prob_thresh', 0.5)
            
            elif base_method == 'pc_gmm':
                method_type = 'pc_gmm'
                gate_params_dict = method_info.get('gate_params')
            
            elif base_method == 'manual':
                method_type = 'manual'
                gate_params_dict = method_info.get('gate_params')
            
            elif base_method == 'auto':
                auto_method_used = method_info.get('auto_method_used', 'pc_gmm')
                if auto_method_used == 'pc_gmm':
                    method_type = 'pc_gmm'
                    gate_params_dict = method_info.get('gate_params')
                elif auto_method_used == 'scoring':
                    method_type = None
                    gate_params_dict = None
            
            elif base_method in ['premessa', 'scoring']:
                method_type = None
                gate_params_dict = None
    
    for idx, gate in enumerate(gates_to_plot):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        if layer is not None and layer in adata.layers:
            gate_idx = adata.var_names.get_loc(gate)
            values = adata.layers[layer][:, gate_idx].flatten()
        else:
            values = adata[:, gate].X.flatten()
        
        if len(values) == 0:
            continue
            
        xmin_auto = np.percentile(values, xlim_percentile[0])
        xmax_auto = np.percentile(values, xlim_percentile[1])
        
        if xlim is not None:
            xmin = xlim[0] if xlim[0] is not None else xmin_auto
            xmax = xlim[1] if xlim[1] is not None else xmax_auto
        else:
            xmin, xmax = xmin_auto, xmax_auto
        
        values_in_range = values[(values >= xmin) & (values <= xmax)]
        
        if len(values_in_range) == 0:
            continue
        
        fig.add_trace(
            go.Histogram(
                x=values_in_range,
                nbinsx=bins,
                name=gate,
                showlegend=False,
                marker=dict(
                    color='#3498db',
                    line=dict(color='#2c3e50', width=0.5)
                ),
                hovertemplate='Intensity: %{x:.2f}<br>Count: %{y}<extra></extra>'
            ),
            row=row,
            col=col
        )
        
        counts, _ = np.histogram(values_in_range, bins=bins)
        ymax = counts.max() if len(counts) > 0 and counts.max() > 0 else 1
        
        # Add method overlays
        if gate_params_dict is not None and method_type is not None:
            gate_idx_in_params = list(barcode_channels).index(gate)
            
            if str(gate_idx_in_params) in gate_params_dict:
                params = gate_params_dict[str(gate_idx_in_params)]
                
                if params is None:
                    continue
                
                # Show OFF/ON means for GMM and PC-GMM
                if method_type in ['gmm', 'pc_gmm']:
                    mu_off = params.get('mu_off')
                    mu_on = params.get('mu_on')
                    
                    if mu_off is not None and xmin <= mu_off <= xmax:
                        fig.add_trace(
                            go.Scatter(
                                x=[mu_off, mu_off],
                                y=[0, ymax],
                                mode='lines',
                                line=dict(color='#e74c3c', dash='dash', width=2),
                                showlegend=False,
                                hovertemplate='<b>OFF mean</b><br>Intensity: %{x:.3f}<extra></extra>',
                                name='OFF'
                            ),
                            row=row,
                            col=col
                        )
                    
                    if mu_on is not None and xmin <= mu_on <= xmax:
                        fig.add_trace(
                            go.Scatter(
                                x=[mu_on, mu_on],
                                y=[0, ymax],
                                mode='lines',
                                line=dict(color='#2ca02c', dash='dash', width=2),
                                showlegend=False,
                                hovertemplate='<b>ON mean</b><br>Intensity: %{x:.3f}<extra></extra>',
                                name='ON'
                            ),
                            row=row,
                            col=col
                        )
                
                # Show threshold for GMM and manual 
                if method_type in ['gmm', 'manual']:
                    threshold = params.get('threshold')
                    
                    if threshold is not None and xmin <= threshold <= xmax:
                        hover_label = 'Threshold' if method_type == 'gmm' else 'Manual Threshold'
                        hover_extra = f' (p={prob_thresh})' if method_type == 'gmm' else ''
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[threshold, threshold],
                                y=[0, ymax],
                                mode='lines',
                                line=dict(color='#f39c12', dash='dot', width=2),
                                showlegend=False,
                                hovertemplate=f'<b>{hover_label}</b>{hover_extra}<br>Intensity: %{{x:.3f}}<extra></extra>',
                                name='Threshold'
                            ),
                            row=row,
                            col=col
                        )
        
        xaxis_type = 'log' if (log_scale_x and not is_transformed) else 'linear'
        yaxis_type = 'log' if log_scale_y else 'linear'
        
        if xaxis_type == 'log':
            x_range = [np.log10(max(xmin, 1e-10)), np.log10(xmax)]
        else:
            x_range = [xmin, xmax]
        
        x_title = "Intensity (log)" if xaxis_type == 'log' else "Intensity"
        y_title = "Count (log)" if yaxis_type == 'log' else "Count"
        
        fig.update_xaxes(
            title_text=x_title,
            type=xaxis_type,
            range=x_range,
            row=row,
            col=col
        )
        
        fig.update_yaxes(
            title_text=y_title,
            type=yaxis_type,
            row=row,
            col=col
        )
    
    title_parts = ["Gate Intensity Distributions"]
    if layer:
        title_parts.append(f"(layer: {layer})")
    if show_method_data and gate_params_dict is not None:
        title_parts.append(f"[{show_method_data}]")
        if method_type == 'gmm':
            title_parts[-1] = f"[{show_method_data}: GMM, threshold={prob_thresh}]"
    
    fig.update_layout(
        title_text=" ".join(title_parts),
        title_font_size=16,
        height=350 * n_rows,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig


def plot_confidence_distribution(adata: ad.AnnData,
                                confidence_col: str,
                                bins: int = 100,
                                show_threshold: bool = True) -> go.Figure:
    """Plot interactive confidence score distribution.
    
    Creates an interactive histogram of confidence scores.
    Shows all thresholds applied to this confidence column.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    confidence_col : str
        Name of confidence column in adata.obs
    bins : int
        Number of histogram bins (default: 100)
    show_threshold : bool
        Show threshold lines if available (default: True)
    
    Returns:
    --------
    fig : plotly Figure
        Interactive figure with hover-enabled histogram
    """
    if confidence_col not in adata.obs.columns:
        raise ValueError(f"Confidence column '{confidence_col}' not found in adata.obs")
    
    confidences = adata.obs[confidence_col].values
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(
            x=confidences,
            nbinsx=bins,
            marker=dict(color='#1f77b4', line=dict(color='black', width=0.5)),
            hovertemplate='Confidence: %{x:.4f}<br>Count: %{y}<extra></extra>',
            name='Confidence'
        )
    )
    
    hist_counts, _ = np.histogram(confidences, bins=bins)
    ymax = hist_counts.max() if len(hist_counts) > 0 else 1
    
    if show_threshold and 'confidence_filtering' in adata.uns:
        filters = []
        for filter_name, filter_info in adata.uns['confidence_filtering'].items():
            if filter_info.get('confidence_col') == confidence_col:
                filters.append((filter_name, filter_info))
        
        for filter_name, filter_info in filters:
            threshold = filter_info.get('threshold')
            method = filter_info.get('method', 'unknown')
            
            if threshold is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[threshold, threshold],
                        y=[0, ymax],
                        mode='lines',
                        line=dict(color='#f39c12', dash='dash', width=2),
                        showlegend=len(filters) > 1,
                        hovertemplate=f'<b>{filter_name}</b><br>Method: {method}<br>Threshold: {threshold:.4f}<extra></extra>',
                        name=f'{filter_name} ({method})'
                    )
                )
                
                if 'gmm_params' in filter_info:
                    gmm_params = filter_info['gmm_params']
                    if 'means' in gmm_params and len(gmm_params['means']) == 2:
                        means = gmm_params['means']
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[means[0], means[0]],
                                y=[0, ymax],
                                mode='lines',
                                line=dict(color='#e74c3c', dash='dot', width=1.5),
                                showlegend=False,
                                hovertemplate=f'<b>{filter_name}</b><br>Low mean: {means[0]:.4f}<extra></extra>',
                                name='Low mean'
                            )
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[means[1], means[1]],
                                y=[0, ymax],
                                mode='lines',
                                line=dict(color='#2ca02c', dash='dot', width=1.5),
                                showlegend=False,
                                hovertemplate=f'<b>{filter_name}</b><br>High mean: {means[1]:.4f}<extra></extra>',
                                name='High mean'
                            )
                        )
    
    fig.update_layout(
        title=f"Confidence Distribution: {confidence_col}",
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        hovermode='closest',
        height=500
    )
    
    return fig



def plot_hamming_graph(adata: ad.AnnData,
                      assignment_col: str,
                      confidence_col: Optional[str] = None,
                      radius: int = 2,
                      min_count: int = 50,
                      valid_only: bool = True,
                      layout: str = 'spring',
                      node_size_min: float = 5.0,
                      node_size_max: float = 200.0,
                      size_transform: str = 'sqrt') -> go.Figure:
    """Plot interactive Hamming graph of barcode patterns.
    
    Creates an interactive network visualization showing barcode patterns
    connected by Hamming distance. Node size represents pattern frequency,
    color represents confidence or validity.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    assignment_col : str
        Column name with barcode assignments (string patterns)
    confidence_col : str, optional
        Column name with confidence scores. If provided, colors nodes by
        mean confidence. If None, colors by validity (green=valid, red=invalid).
    radius : int
        Exact Hamming distance for edges (default: 2)
    min_count : int
        Minimum count for patterns to include (default: 50)
    valid_only : bool
        Only show valid patterns (default: True)
    layout : str
        Layout algorithm: 'spring', 'circular', 'kamada_kawai' (default: 'spring')
    node_size_min : float
        Minimum node size (default: 5.0)
    node_size_max : float
        Maximum node size (default: 200.0)
    size_transform : str
        Size transformation: 'sqrt', 'log', or 'linear' (default: 'sqrt')
    
    Returns:
    --------
    fig : plotly Figure
        Interactive network graph with zoom/pan
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx is required for Hamming graph visualization. "
            "Install it with: pip install networkx"
        )
    
    from .barcode import is_valid_pattern
    
    if assignment_col not in adata.obs.columns:
        raise ValueError(f"Assignment column '{assignment_col}' not found in adata.obs")
    
    if confidence_col is not None and confidence_col not in adata.obs.columns:
        raise ValueError(f"Confidence column '{confidence_col}' not found in adata.obs")
    
    patterns_str = adata.obs[assignment_col].values
    confidences = adata.obs[confidence_col].values if confidence_col else None
    
    from collections import defaultdict
    pattern_data = defaultdict(lambda: {'count': 0, 'confidences': []})
    
    for i, pattern_str in enumerate(patterns_str):
        pattern = tuple(int(c) for c in pattern_str)
        pattern_data[pattern]['count'] += 1
        if confidences is not None:
            pattern_data[pattern]['confidences'].append(confidences[i])
    
    if valid_only:
        pattern_data = {p: d for p, d in pattern_data.items() if is_valid_pattern(p)}
    
    pattern_data = {p: d for p, d in pattern_data.items() if d['count'] >= min_count}
    
    if len(pattern_data) == 0:
        raise ValueError(f"No patterns found with min_count={min_count}" + 
                        (" and valid_only=True" if valid_only else ""))
    
    print(f"Building Hamming graph with {len(pattern_data)} patterns...")
    
    G = nx.Graph()
    
    for pattern, data in pattern_data.items():
        G.add_node(
            pattern,
            count=data['count'],
            valid=is_valid_pattern(pattern),
            confidence=np.mean(data['confidences']) if data['confidences'] else None
        )
    
    patterns_list = list(pattern_data.keys())
    for i in range(len(patterns_list)):
        for j in range(i + 1, len(patterns_list)):
            if sum(a != b for a, b in zip(patterns_list[i], patterns_list[j])) == radius:
                G.add_edge(patterns_list[i], patterns_list[j])
    
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    if G.number_of_nodes() == 0:
        raise ValueError("No nodes in graph after filtering")
    
    n_nodes = G.number_of_nodes()
    
    if layout == 'spring':
        k = 1.0 / np.sqrt(n_nodes) if n_nodes > 10 else 0.5
        pos = nx.spring_layout(G, seed=42, k=k, iterations=100)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError(f"Unknown layout: {layout}")
    
    scale = 10 * np.sqrt(n_nodes)
    pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='#888'),
        opacity=0.3,
        hoverinfo='none',
        showlegend=False
    )
    
    counts = [G.nodes[n]['count'] for n in G.nodes()]
    
    transform_fn = {'sqrt': np.sqrt, 'log': np.log1p, 'linear': lambda x: x}[size_transform]
    transformed = [transform_fn(c) for c in counts]
    t_min, t_max = min(transformed), max(transformed)
    
    node_x, node_y, node_sizes, node_colors, node_borders, node_text = [], [], [], [], [], []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        count = G.nodes[node]['count']
        valid = G.nodes[node]['valid']
        conf = G.nodes[node]['confidence']
        
        t = transform_fn(count)
        size = node_size_min + (t - t_min) / (t_max - t_min) * (node_size_max - node_size_min) if t_max > t_min else (node_size_min + node_size_max) / 2
        node_sizes.append(size)
        
        node_colors.append(conf if conf is not None else ('#2ca02c' if valid else '#d62728'))
        node_borders.append('#000000' if valid else '#888888')
        
        pattern_set = {i+1 for i, b in enumerate(node) if b == 1}
        hover = f"Pattern: {pattern_set}<br>Count: {count}<br>Valid: {valid}"
        if conf is not None:
            hover += f"<br>Mean confidence: {conf:.4f}"
        node_text.append(hover)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale='RdYlGn' if confidence_col else None,
            showscale=confidence_col is not None,
            colorbar=dict(title='Mean<br>Confidence', thickness=15, len=0.7) if confidence_col else None,
            line=dict(width=2, color=node_borders),
            cmin=0, cmax=1
        ),
        text=node_text,
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    
    title = f"Hamming Graph (radius={radius}, min_count={min_count})"
    if valid_only:
        title += " - Valid patterns only"
    
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        plot_bgcolor='white'
    )
    
    return fig


def plot_hamming_heatmap(adata: ad.AnnData,
                         assignment_col: str,
                         hamming_radius: int = 2,
                         valid_only: bool = True,
                         min_count: int = 50,
                         cluster_patterns: bool = True,
                         use_rank_labels: bool = True) -> go.Figure:
    """Plot Hamming distance heatmap for barcode patterns.
    
    Creates a heatmap showing pairwise Hamming distances between patterns.
    Patterns can be clustered to group similar patterns together.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    assignment_col : str
        Column name with barcode assignments (string patterns)
    hamming_radius : int
        Maximum Hamming distance to display (default: 2).
        Distances > radius shown as white.
    valid_only : bool
        Only show valid patterns (default: True)
    min_count : int
        Minimum count for patterns to include (default: 50)
    cluster_patterns : bool
        Cluster patterns by Hamming distance (default: True)
    use_rank_labels : bool
        Use rank numbers (#1, #2, ...) instead of full pattern on axes (default: True).
        Makes heatmap more readable with many patterns.
    
    Returns:
    --------
    fig : plotly Figure
        Interactive heatmap
    """
    from collections import Counter
    from .barcode import is_valid_pattern
    
    if assignment_col not in adata.obs.columns:
        raise ValueError(f"Assignment column '{assignment_col}' not found in adata.obs")
    
    patterns_str = adata.obs[assignment_col].values
    patterns = [tuple(int(c) for c in p) for p in patterns_str]
    pattern_counts = Counter(patterns)
    
    if valid_only:
        pattern_counts = {p: c for p, c in pattern_counts.items() if is_valid_pattern(p)}
    
    pattern_counts = {p: c for p, c in pattern_counts.items() if c >= min_count}
    
    if len(pattern_counts) == 0:
        raise ValueError(f"No patterns found with min_count={min_count}" + 
                        (" and valid_only=True" if valid_only else ""))
    
    if len(pattern_counts) < 2:
        raise ValueError("Need at least 2 patterns for heatmap")
    
    patterns_list = [p for p, _ in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)]
    n = len(patterns_list)
    
    print(f"Computing Hamming distance matrix for {n} patterns...")
    
    dist_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i, n):
            dist = 0 if i == j else sum(a != b for a, b in zip(patterns_list[i], patterns_list[j]))
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    
    if cluster_patterns and n > 2:
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
            
            order = leaves_list(linkage(squareform(dist_matrix), method='average'))
            patterns_list = [patterns_list[i] for i in order]
            dist_matrix = dist_matrix[order, :][:, order]
            print(f"  Patterns clustered by hierarchical clustering")
        except ImportError:
            print(f"  scipy not available, skipping clustering")
    
    labels = [f"#{i+1}" if use_rank_labels else f"#{i+1}: {{{', '.join(str(j+1) for j, b in enumerate(p) if b)}}}" 
              for i, p in enumerate(patterns_list)]
    
    hover = []
    for i in range(n):
        row = []
        for j in range(n):
            pi = {k+1 for k, b in enumerate(patterns_list[i]) if b}
            pj = {k+1 for k, b in enumerate(patterns_list[j]) if b}
            row.append(
                f"Pattern A (#{i+1}): {pi}<br>"
                f"Pattern B (#{j+1}): {pj}<br>"
                f"Hamming distance: {dist_matrix[i, j]}<br>"
                f"Count A: {pattern_counts[patterns_list[i]]}<br>"
                f"Count B: {pattern_counts[patterns_list[j]]}"
            )
        hover.append(row)
    
    display = dist_matrix.astype(float)
    display[display > hamming_radius] = np.nan
    
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=display,
            x=labels, y=labels,
            colorscale=[
                [0.0, '#ffffff'], [0.2, '#e8f4f8'], [0.4, '#a8d8ea'],
                [0.6, '#6eb5d0'], [0.8, '#3d8fb3'], [1.0, '#1e5a7d']
            ],
            zmid=hamming_radius / 2,
            zmin=0, zmax=hamming_radius,
            text=hover,
            hovertemplate='%{text}<extra></extra>',
            colorbar=dict(
                title="Hamming<br>Distance",
                thickness=15, len=0.7,
                tickmode='linear', tick0=0, dtick=1
            )
        )
    )
    
    info = [f"max_distance={hamming_radius}"]
    if valid_only:
        info.append("Valid patterns only")
    if cluster_patterns:
        info.append("Hierarchically clustered")
    info.append(f"{n} patterns, min_count={min_count}")
    
    fig.update_layout(
        title=dict(
            text=f"Hamming Distance Heatmap<br><sub>{' | '.join(info)}</sub>",
            x=0.5, xanchor='center'
        ),
        xaxis=dict(title="", tickangle=-45 if not use_rank_labels else 0, side='bottom'),
        yaxis=dict(title="", autorange='reversed'),
        height=max(600, n * 30),
        width=max(700, n * 30),
        hovermode='closest'
    )
    
    return fig