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
    "plot_channel_intensities", 
    "plot_intensity_scatter",
    "plot_cumul_barcode_rank"
    "plot_confidence_distribution", 
    "plot_hamming_graph",
    "plot_barcode_rank_histogram",
    "plot_hamming_heatmap"
]

def plot_intensity_scatter(adata: ad.AnnData,
                           layer: Optional[str] = 'log',
                           method: Optional[str] = None,
                           percentile: float = 99.0,
                           sum_low: Optional[float] = 1.0,
                           sum_high: Optional[float] = 99.0,
                           var_low: Optional[float] = 1.0,
                           var_high: Optional[float] = 99.0,
                           subsample: Optional[int] = 50000) -> go.Figure:
    """Plot channel sum vs variance scatter with optional filter preview.
    
    Helps visualize cell distribution and preview intensity filtering
    before applying it.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with mapped barcode channels.
    layer : str, optional, default 'log'
        Data layer to use. None for ``adata.X``.
    method : {'rectangular', 'ellipsoidal'}, optional
        Filter method to preview. If None, shows raw scatter without filtering.
    percentile : float, default 99.0
        For ``'ellipsoidal'`` method, percentile threshold.
    sum_low : float, optional, default 1.0
        For ``'rectangular'``: lower percentile bound for channel sum.
    sum_high : float, optional, default 99.0
        For ``'rectangular'``: upper percentile bound for channel sum.
    var_low : float, optional, default 1.0
        For ``'rectangular'``: lower percentile bound for channel variance.
    var_high : float, optional, default 99.0
        For ``'rectangular'``: upper percentile bound for channel variance.
    subsample : int, optional, default 50000
        Max cells to plot (for performance). None for all.
    
    Returns
    -------
    Figure
        Interactive Plotly scatter plot.
    """
    if method is not None and method not in {'rectangular', 'ellipsoidal'}:
        raise ValueError(f"method must be 'rectangular', 'ellipsoidal', or None, got {method!r}")
    if 'barcode_channels' not in adata.uns:
        raise ValueError("Barcode channels not mapped. Run map_channels first.")
    
    bc_channels = adata.uns['barcode_channels']
    bc_idx = [list(adata.var_names).index(ch) for ch in bc_channels]
    X = adata.layers[layer][:, bc_idx] if layer else adata.X[:, bc_idx]
    
    channel_sum = np.abs(X).sum(axis=1)
    channel_var = np.var(X, axis=1)
    n_total = len(adata)
    
    # Subsample for plotting
    if subsample is not None and n_total > subsample:
        np.random.seed(42)
        idx = np.random.choice(n_total, subsample, replace=False)
        channel_sum_plot = channel_sum[idx]
        channel_var_plot = channel_var[idx]
        subsampled = True
    else:
        channel_sum_plot = channel_sum
        channel_var_plot = channel_var
        idx = np.arange(n_total)
        subsampled = False
    
    fig = go.Figure()
    
    if method is None:
        # Raw scatter
        fig.add_trace(
            go.Scattergl(
                x=channel_sum_plot,
                y=channel_var_plot,
                mode='markers',
                marker=dict(size=3, color='#3498db', opacity=0.3),
                hovertemplate='Sum: %{x:.2f}<br>Var: %{y:.4f}<extra></extra>',
                showlegend=False
            )
        )
        
        subtitle = f"{len(channel_sum_plot):,} cells"
        if subsampled:
            subtitle += f" (subsampled from {n_total:,})"
        
        fig.update_layout(
            title=f"Channel Sum vs Variance<br><sub>{subtitle}</sub>",
        )
    
    else:
        if method == 'rectangular':
            thresholds = {
                'sum_low': np.percentile(channel_sum, sum_low) if sum_low is not None else None,
                'sum_high': np.percentile(channel_sum, sum_high) if sum_high is not None else None,
                'var_low': np.percentile(channel_var, var_low) if var_low is not None else None,
                'var_high': np.percentile(channel_var, var_high) if var_high is not None else None,
            }
            
            pass_mask = np.ones(n_total, dtype=bool)
            if thresholds['sum_low'] is not None:
                pass_mask &= channel_sum >= thresholds['sum_low']
            if thresholds['sum_high'] is not None:
                pass_mask &= channel_sum <= thresholds['sum_high']
            if thresholds['var_low'] is not None:
                pass_mask &= channel_var >= thresholds['var_low']
            if thresholds['var_high'] is not None:
                pass_mask &= channel_var <= thresholds['var_high']
            
            param_str = f"sum=[{sum_low}, {sum_high}], var=[{var_low}, {var_high}]"
        
        else:  # ellipsoidal
            from sklearn.covariance import MinCovDet
            
            data = np.column_stack([channel_sum, channel_var])
            mcd = MinCovDet(random_state=42, support_fraction=0.75)
            mcd.fit(data)
            mahal_dist = np.sqrt(mcd.mahalanobis(data))
            threshold = np.percentile(mahal_dist, percentile)
            pass_mask = mahal_dist <= threshold
            
            param_str = f"percentile={percentile}"
        
        pass_mask_plot = pass_mask[idx]
        fail_mask_plot = ~pass_mask_plot
        
        n_pass = pass_mask.sum()
        n_fail = n_total - n_pass
        pct_pass = 100 * n_pass / n_total
        
        fig.add_trace(
            go.Scatter(
                x=channel_sum_plot[fail_mask_plot],
                y=channel_var_plot[fail_mask_plot],
                mode='markers',
                marker=dict(size=3, color='#e74c3c', opacity=0.5),
                name=f'Removed ({n_fail:,})',
                hovertemplate='Sum: %{x:.2f}<br>Var: %{y:.4f}<extra>Removed</extra>',
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=channel_sum_plot[pass_mask_plot],
                y=channel_var_plot[pass_mask_plot],
                mode='markers',
                marker=dict(size=3, color='#3498db', opacity=0.3),
                name=f'Kept ({n_pass:,})',
                hovertemplate='Sum: %{x:.2f}<br>Var: %{y:.4f}<extra>Kept</extra>',
            )
        )
        
        subtitle_parts = [f"{method}", param_str, f"{n_pass:,}/{n_total:,} kept ({pct_pass:.1f}%)"]
        if subsampled:
            subtitle_parts.append(f"showing {len(channel_sum_plot):,}")
        
        fig.update_layout(
            title=f"Intensity Filter Preview<br><sub>{' | '.join(subtitle_parts)}</sub>",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
    
    fig.update_xaxes(title_text="Channel Sum")
    fig.update_yaxes(title_text="Channel Variance")
    fig.update_layout(
        height=500,
        hovermode='closest'
    )
    
    return fig


def plot_barcode_rank_histogram(adata: ad.AnnData,
                                assignment_col: str,
                                confidence_col: Optional[str] = None,
                                metric: str = 'count',
                                top_n: int = 20,
                                min_metric: float = 0,
                                valid_only: bool = False) -> go.Figure:
    """Plot rank histogram of top barcode patterns.
    
    Shows the most frequent barcode patterns as a ranked bar chart.
    Valid patterns shown in blue, invalid patterns in grey.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    assignment_col : str
        Column name with barcode assignments (string patterns).
    confidence_col : str, optional
        Column name with confidence scores. Required for ``metric='median_conf'``
        or ``'score'``.
    metric : {'count', 'median_conf', 'score'}, default 'count'
        Metric to rank and display.
    top_n : int, default 20
        Number of top patterns to show.
    min_metric : float, default 0
        Minimum metric value to include pattern.
    valid_only : bool, default False
        Only show valid patterns.
    
    Returns
    -------
    Figure
        Interactive Plotly bar chart.
    """
    from .barcode import is_valid_pattern
    
    if assignment_col not in adata.obs.columns:
        raise ValueError(f"Assignment column '{assignment_col}' not found in adata.obs")
    if metric not in {'count', 'median_conf', 'score'}:
        raise ValueError(f"metric must be 'count', 'median_conf', or 'score', got {metric!r}")
    if metric in {'median_conf', 'score'} and confidence_col is None:
        raise ValueError(f"confidence_col required for metric='{metric}'")
    if confidence_col is not None and confidence_col not in adata.obs.columns:
        raise ValueError(f"Confidence column '{confidence_col}' not found in adata.obs")
    
    patterns_str = adata.obs[assignment_col].values.astype(str)
    confidences = adata.obs[confidence_col].values.astype(np.float64) if confidence_col else None
    
    pattern_to_cells = {}
    for i, p in enumerate(patterns_str):
        if p not in pattern_to_cells:
            pattern_to_cells[p] = []
        pattern_to_cells[p].append(i)
    
    pattern_stats = {}
    for p, indices in pattern_to_cells.items():
        pattern_tuple = tuple(int(c) for c in p)
        if valid_only and not is_valid_pattern(pattern_tuple):
            continue
        count = len(indices)
        median_conf = float(np.median(confidences[indices])) if confidences is not None else None
        score = count * median_conf if median_conf is not None else None
        
        metric_value = {'count': count, 'median_conf': median_conf, 'score': score}[metric]
        if metric_value is None or metric_value < min_metric:
            continue
        
        pattern_stats[p] = {
            'count': count,
            'median_conf': median_conf,
            'score': score,
            'pattern_tuple': pattern_tuple
        }
    
    if not pattern_stats:
        raise ValueError(f"No patterns with {metric} >= {min_metric}" +
                        (" and valid_only=True" if valid_only else ""))
    
    # Sort by metric
    top_patterns = sorted(pattern_stats.items(), key=lambda x: x[1][metric], reverse=True)[:top_n]
    
    ranks = list(range(1, len(top_patterns) + 1))
    values = [stats[metric] for _, stats in top_patterns]
    counts = [stats['count'] for _, stats in top_patterns]
    colors = ['#3498db' if is_valid_pattern(stats['pattern_tuple']) else '#95a5a6' 
              for _, stats in top_patterns]
    
    hover_texts = []
    for r, (p, stats) in enumerate(top_patterns, 1):
        pt = stats['pattern_tuple']
        text = (f"Rank: {r}<br>"
                f"Pattern: {{{', '.join(str(i+1) for i, b in enumerate(pt) if b)}}}<br>"
                f"Count: {stats['count']}<br>"
                f"Valid: {is_valid_pattern(pt)}")
        if stats['median_conf'] is not None:
            text += f"<br>Median conf: {stats['median_conf']:.3f}"
            text += f"<br>Score: {stats['score']:.1f}"
        hover_texts.append(text)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=ranks,
            y=values,
            marker=dict(color=colors, line=dict(color='#2c3e50', width=1)),
            customdata=hover_texts,
            hovertemplate='%{customdata}<extra></extra>',
            showlegend=False
        )
    )
    
    metric_labels = {
        'count': 'Count',
        'median_conf': 'Median Confidence',
        'score': 'Score (count × median_conf)'
    }
    
    subtitle_parts = [f"Top {len(top_patterns)} patterns by {metric}",
                      f"{sum(counts):,} cells",
                      f"min_{metric}={min_metric}"]
    if valid_only:
        subtitle_parts.append("valid only")
    
    fig.update_layout(
        title=f"Barcode Pattern Rank Histogram<br>"
              f"<sub>{' | '.join(subtitle_parts)}</sub>",
        xaxis=dict(
            title=f"Rank (by {metric})",
            tickmode='linear',
            tick0=1,
            dtick=1 if len(top_patterns) <= 20 else 5
        ),
        yaxis_title=metric_labels[metric],
        hovermode='closest',
        height=500
    )
    
    return fig


def plot_cumul_barcode_rank(adata: ad.AnnData,
                            assignment_col: str,
                            confidence_col: Optional[str] = None,
                            metric: str = 'count',
                            valid_only: bool = False,
                            normalize: bool = True,
                            log_x: bool = True) -> go.Figure:
    """Plot cumulative barcode rank curve.
    
    Shows cumulative sum of metric values across patterns ranked by that metric.
    Useful for visualizing how many top patterns capture most of the data.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    assignment_col : str
        Column name with barcode assignments (string patterns).
    confidence_col : str, optional
        Column name with confidence scores. Required for ``metric='score'``.
    metric : {'count', 'score'}, default 'count'
        Metric to rank by: ``'count'`` or ``'score'`` (count * median_conf).
    valid_only : bool, default False
        Only include valid patterns.
    normalize : bool, default True
        Normalize cumulative sum to [0, 1].
    log_x : bool, default True
        Use log scale for x-axis.
    
    Returns
    -------
    Figure
        Interactive Plotly cumulative rank curve.
    """
    from .barcode import is_valid_pattern
    
    if assignment_col not in adata.obs.columns:
        raise ValueError(f"Assignment column '{assignment_col}' not found in adata.obs")
    if metric not in {'count', 'score'}:
        raise ValueError(f"metric must be 'count' or 'score', got {metric!r}")
    if metric == 'score' and confidence_col is None:
        raise ValueError("confidence_col required for metric='score'")
    if confidence_col is not None and confidence_col not in adata.obs.columns:
        raise ValueError(f"Confidence column '{confidence_col}' not found in adata.obs")
    
    patterns_str = adata.obs[assignment_col].values.astype(str)
    confidences = adata.obs[confidence_col].values.astype(np.float64) if confidence_col else None
    
    pattern_to_cells = {}
    for i, p in enumerate(patterns_str):
        if p not in pattern_to_cells:
            pattern_to_cells[p] = []
        pattern_to_cells[p].append(i)
    
    metric_values = []
    for p, indices in pattern_to_cells.items():
        pattern_tuple = tuple(int(c) for c in p)
        if valid_only and not is_valid_pattern(pattern_tuple):
            continue
        count = len(indices)
        if metric == 'score':
            median_conf = float(np.median(confidences[indices]))
            metric_values.append(count * median_conf)
        else:
            metric_values.append(count)
    
    if not metric_values:
        raise ValueError(f"No patterns found" + (" with valid_only=True" if valid_only else ""))
    
    metric_values = np.array(sorted(metric_values, reverse=True))
    cumsum = np.cumsum(metric_values)
    if normalize:
        cumsum = cumsum / cumsum[-1]
    
    ranks = np.arange(1, len(metric_values) + 1)
    
    hover_texts = [f"Rank: {r}<br>{metric}: {v:.1f}<br>Cumulative: {c:.4f}" 
                   for r, v, c in zip(ranks, metric_values, cumsum)]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=ranks,
            y=cumsum,
            mode='lines',
            line=dict(color='#3498db', width=2),
            customdata=hover_texts,
            hovertemplate='%{customdata}<extra></extra>',
            showlegend=False
        )
    )
    
    y_label = f"Cumulative {metric}" + (" (normalized)" if normalize else "")
    
    subtitle_parts = [f"{len(metric_values)} patterns"]
    if valid_only:
        subtitle_parts.append("valid only")
    
    fig.update_layout(
        title=f"Cumulative Barcode Rank Curve<br><sub>{' | '.join(subtitle_parts)}</sub>",
        xaxis_title="Rank (by " + metric + ")",
        yaxis_title=y_label,
        hovermode='closest',
        height=500
    )
    
    if log_x:
        fig.update_xaxes(type="log")
    
    return fig


def plot_channel_intensities(adata: ad.AnnData,
                         channels: Optional[List[str]] = None,
                         layer: Optional[str] = None,
                         show_method_data: Optional[str] = None,
                         log_scale_x: bool = True,
                         log_scale_y: bool = False,
                         bins: int = 250,
                         xlim: Optional[tuple] = None,
                         xlim_percentile: tuple = (0.1, 99.9)) -> go.Figure:
    """Plot interactive channel intensity distributions.
    
    Creates interactive histograms for barcode channel intensities.
    Can overlay learned parameters from debarcoding methods.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    channels : list of str, optional
        Specific channels to visualize. If None, shows all barcode channels.
    layer : str, optional
        Layer to use for visualization. If None, uses ``.X`` (raw data).
        If using a transformed layer (e.g., ``'log'``, ``'arcsinh_cf5'``), 
        set ``log_scale_x=False`` since data is already transformed.
    show_method_data : str, optional
        Name of debarcoding result from ``adata.uns['debarcoding']`` to visualize.
        Shows overlays based on the method (GMM means, thresholds, etc.).
    log_scale_x : bool, default True
        Use log scale for x-axis. Set to False when using transformed layers.
    log_scale_y : bool, default False
        Use log scale for y-axis.
    bins : int, default 250
        Number of histogram bins.
    xlim : tuple, optional
        X-axis limits as ``(xmin, xmax)``. Supports partial specification
        with None for automatic limits.
    xlim_percentile : tuple, default (0.1, 99.9)
        Percentile range for automatic limits.
    
    Returns
    -------
    Figure
        Interactive Plotly figure with channel histograms.
    """
    
    from .io import get_barcode_channels
    barcode_channels = get_barcode_channels(adata)
    
    if channels is None:
        channels_to_plot = barcode_channels
    else:
        invalid = set(channels) - set(barcode_channels)
        if invalid:
            raise ValueError(f"Invalid channels (not barcode channels): {invalid}")
        channels_to_plot = channels
    
    n_channels = len(channels_to_plot)
    n_cols = min(6, n_channels)
    n_rows = int(np.ceil(n_channels / n_cols))
    
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
        subplot_titles=channels_to_plot,
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    method_type = None
    channel_params_dict = None
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
                channel_params_dict = method_info.get('channel_params')
                prob_thresh = method_info.get('prob_thresh', 0.5)
            
            elif base_method == 'pc_gmm':
                method_type = 'pc_gmm'
                channel_params_dict = method_info.get('channel_params')
            
            elif base_method == 'manual':
                method_type = 'manual'
                channel_params_dict = method_info.get('channel_params')
            
            elif base_method == 'auto':
                auto_method_used = method_info.get('auto_method_used', 'pc_gmm')
                if auto_method_used == 'pc_gmm':
                    method_type = 'pc_gmm'
                    channel_params_dict = method_info.get('channel_params')
                elif auto_method_used == 'scoring':
                    method_type = None
                    channel_params_dict = None
            
            elif base_method in ['premessa', 'scoring']:
                method_type = None
                channel_params_dict = None
    
    for idx, channel in enumerate(channels_to_plot):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        if layer is not None and layer in adata.layers:
            channel_idx = adata.var_names.get_loc(channel)
            values = adata.layers[layer][:, channel_idx].flatten()
        else:
            values = adata[:, channel].X.flatten()
        
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
                name=channel,
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
        if channel_params_dict is not None and method_type is not None:
            channel_idx_in_params = list(barcode_channels).index(channel)
            
            if str(channel_idx_in_params) in channel_params_dict:
                params = channel_params_dict[str(channel_idx_in_params)]
                
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
    
    title_parts = ["Channel Intensity Distributions"]
    if layer:
        title_parts.append(f"(layer: {layer})")
    if show_method_data and channel_params_dict is not None:
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
                                show_threshold: str = None) -> go.Figure:
    """Plot interactive confidence score distribution.
    
    Creates an interactive histogram of confidence scores.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    confidence_col : str
        Name of confidence column in ``adata.obs``.
    bins : int, default 100
        Number of histogram bins.
    show_threshold : str, optional
        Filter name to display threshold line for.
    
    Returns
    -------
    Figure
        Interactive Plotly histogram.
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
        if show_threshold in adata.uns['confidence_filtering']:
            filter_info = adata.uns['confidence_filtering'][show_threshold]
            
            if filter_info.get('confidence_col') == confidence_col:
                threshold = filter_info.get('threshold')
                method = filter_info.get('method', 'unknown')
                
                if threshold is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[threshold, threshold],
                            y=[0, ymax],
                            mode='lines',
                            line=dict(color='#f39c12', dash='dash', width=2),
                            showlegend=True,
                            hovertemplate=f'<b>{show_threshold}</b><br>Method: {method}<br>Threshold: {threshold:.4f}<extra></extra>',
                            name=f'{show_threshold} ({method})'
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
                                    hovertemplate=f'<b>{show_threshold}</b><br>Low mean: {means[0]:.4f}<extra></extra>',
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
                                    hovertemplate=f'<b>{show_threshold}</b><br>High mean: {means[1]:.4f}<extra></extra>',
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
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    assignment_col : str
        Column name with barcode assignments (string patterns).
    confidence_col : str, optional
        Column name with confidence scores. If provided, colors nodes by
        mean confidence. If None, colors by validity (green=valid, red=invalid).
    radius : int, default 2
        Exact Hamming distance for edges.
    min_count : int, default 50
        Minimum count for patterns to include.
    valid_only : bool, default True
        Only show valid patterns.
    layout : {'spring', 'circular', 'kamada_kawai'}, default 'spring'
        Layout algorithm.
    node_size_min : float, default 5.0
        Minimum node size.
    node_size_max : float, default 200.0
        Maximum node size.
    size_transform : {'sqrt', 'log', 'linear'}, default 'sqrt'
        Size transformation for node counts.
    
    Returns
    -------
    Figure
        Interactive Plotly network graph.
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
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    assignment_col : str
        Column name with barcode assignments (string patterns).
    hamming_radius : int, default 2
        Maximum Hamming distance to display. Distances > radius shown as white.
    valid_only : bool, default True
        Only show valid patterns.
    min_count : int, default 50
        Minimum count for patterns to include.
    cluster_patterns : bool, default True
        Cluster patterns by Hamming distance.
    use_rank_labels : bool, default True
        Use rank numbers (#1, #2, ...) instead of full pattern on axes.
        Makes heatmap more readable with many patterns.
    
    Returns
    -------
    Figure
        Interactive Plotly heatmap.
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
    pattern_ranks = {p: i+1 for i, p in enumerate(patterns_list)}  # ← Store original ranks
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
    
    labels = [f"#{pattern_ranks[p]}" if use_rank_labels else f"#{pattern_ranks[p]}: {{{', '.join(str(j+1) for j, b in enumerate(p) if b)}}}" 
              for p in patterns_list]  
    
    hover = []
    for i in range(n):
        row = []
        for j in range(n):
            pi = {k+1 for k, b in enumerate(patterns_list[i]) if b}
            pj = {k+1 for k, b in enumerate(patterns_list[j]) if b}
            row.append(
                f"Pattern A (#{pattern_ranks[patterns_list[i]]}): {pi}<br>"  
                f"Pattern B (#{pattern_ranks[patterns_list[j]]}): {pj}<br>"  
                f"Hamming distance: {dist_matrix[i, j]}<br>"
                f"Count A: {pattern_counts[patterns_list[i]]}<br>"
                f"Count B: {pattern_counts[patterns_list[j]]}"
            )
        hover.append(row)
    
    display = dist_matrix.astype(float)
    grey_value = hamming_radius + 1
    display[display > hamming_radius] = grey_value
    
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=display,
            x=labels, y=labels,
            colorscale=[
                [0.0, '#ffffff'], 
                [0.2 * hamming_radius/grey_value, '#e8f4f8'], 
                [0.4 * hamming_radius/grey_value, '#a8d8ea'],
                [0.6 * hamming_radius/grey_value, '#6eb5d0'], 
                [0.8 * hamming_radius/grey_value, '#3d8fb3'], 
                [hamming_radius/grey_value, '#1e5a7d'],
                [hamming_radius/grey_value + 0.001, '#e5ecf6'], 
                [1.0, '#e5ecf6']
            ],
            zmid=hamming_radius / 2,
            zmin=0, zmax=grey_value,
            xgap=1, 
            ygap=1,  
            text=hover,
            hovertemplate='%{text}<extra></extra>',
            colorbar=dict(
                title="Hamming<br>Distance",
                thickness=15, len=0.7,
                tickmode='array',  
                tickvals=list(range(hamming_radius + 1)) + [grey_value],  
                ticktext=list(range(hamming_radius + 1)) + ['Higher'] 
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
        hovermode='closest',
        plot_bgcolor='white' 
    )
    
    return fig