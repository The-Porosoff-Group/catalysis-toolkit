"""Publication-quality XRD refinement and candidate-preview figures."""

import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from .presentation import (
    clean_descriptive_text,
    enrich_phase_results,
    format_chemical_formula,
    format_wavelength_label,
    phase_legend_label,
)


# Existing UI palette, retained for dark figures and candidate previews.
PHASE_COLORS = [
    '#f78166',  # coral red
    '#56d364',  # green
    '#e3b341',  # amber
    '#bc8cff',  # purple
    '#79c0ff',  # sky blue
    '#ffa657',  # orange
    '#ff7eb6',  # pink
    '#7ee787',  # mint
]

# Colorblind-accessible palette with sufficient contrast on a white page.
PUBLICATION_PHASE_COLORS = [
    '#0072b2',  # blue
    '#d55e00',  # vermilion
    '#009e73',  # bluish green
    '#cc79a7',  # reddish purple
    '#e69f00',  # orange
    '#56b4e9',  # sky blue
    '#7b61a8',  # violet
    '#8c6d31',  # brown
]

PLOT_THEMES = {
    'light': {
        'figure': '#ffffff', 'surface': '#ffffff', 'grid': '#d9dee7',
        'text': '#111827', 'muted': '#596273', 'observed': '#111827',
        'calculated': '#c1121f', 'residual': '#1d4ed8',
        'zero': '#4b5563', 'stats_face': '#f7f8fa',
        'phase_colors': PUBLICATION_PHASE_COLORS,
    },
    'dark': {
        'figure': '#0d1117', 'surface': '#161b22', 'grid': '#39414d',
        'text': '#f0f3f6', 'muted': '#a7b0bd', 'observed': '#e6edf3',
        'calculated': '#ff7b72', 'residual': '#79c0ff',
        'zero': '#8b949e', 'stats_face': '#1c2128',
        'phase_colors': PHASE_COLORS,
    },
}


def make_xrd_plot(result, metadata, output_path, theme=None):
    """Render a fitted XRD pattern at its intended publication dimensions.

    Parameters
    ----------
    result : dict
        Completed refinement result.  Its numerical arrays are read-only here.
    metadata : dict
        Sample, wavelength, method, and optional ``plot_theme`` information.
    output_path : str
        Destination PNG path.
    theme : {"light", "dark"}, optional
        Explicit figure theme.  Light is the publication default.
    """
    theme = str(theme or metadata.get('plot_theme') or 'light').lower()
    if theme not in PLOT_THEMES:
        theme = 'light'
    palette = PLOT_THEMES[theme]

    # This enriches labels and tick metadata only.  Fit arrays and statistics
    # are not changed.
    enrich_phase_results(result)

    tt = np.asarray(result['tt'], dtype=float)
    y_obs = np.asarray(result['y_obs'], dtype=float)
    y_calc = np.asarray(result['y_calc'], dtype=float)
    y_bg = np.asarray(result['y_background'], dtype=float)
    resid = np.asarray(result['residuals'], dtype=float)
    phases = result.get('phase_results', []) or []
    phase_patterns = result.get('phase_patterns', []) or []
    n_phases = len(phases)
    stats = result['statistics']

    # Author at the target 6.5-inch manuscript width.  This avoids the former
    # 13-inch canvas being shrunk by half (and shrinking 7–9 point text with it).
    legend_count = 3 + n_phases
    legend_columns = 2 if legend_count > 3 else max(legend_count, 1)
    legend_rows = max(1, math.ceil(legend_count / legend_columns))
    tick_height = max(1.25, 0.78 * max(n_phases, 1))
    height_ratios = [4.6, tick_height, 1.45]
    figure_height = 4.85 + tick_height + 0.28 * legend_rows

    fig = plt.figure(figsize=(6.5, figure_height),
                     facecolor=palette['figure'])
    grid = gridspec.GridSpec(
        3, 1, figure=fig, hspace=0.08, height_ratios=height_ratios)
    ax_main = fig.add_subplot(grid[0])
    ax_ticks = fig.add_subplot(grid[1], sharex=ax_main)
    ax_res = fig.add_subplot(grid[2], sharex=ax_main)

    background = palette['figure']
    surface = palette['surface']
    grid_color = palette['grid']
    text_color = palette['text']
    muted_color = palette['muted']
    phase_colors = palette['phase_colors']

    def style_axis(axis, show_xlabel=False, show_grid=True):
        axis.set_facecolor(surface)
        axis.tick_params(
            colors=text_color, labelsize=9, width=0.8, length=3.5,
            labelbottom=show_xlabel)
        axis.xaxis.label.set_color(text_color)
        axis.yaxis.label.set_color(text_color)
        for spine in axis.spines.values():
            spine.set_edgecolor(grid_color)
            spine.set_linewidth(0.75)
        if show_grid:
            axis.grid(True, color=grid_color, alpha=0.55, linewidth=0.55)
        else:
            axis.grid(False)

    style_axis(ax_main)
    style_axis(ax_ticks, show_grid=False)
    style_axis(ax_res, show_xlabel=True)

    # Shaded, stacked fitted components sit behind the precise total fit line.
    cumulative = np.array(y_bg, dtype=float)
    for index, pattern in enumerate(phase_patterns[:n_phases]):
        color = phase_colors[index % len(phase_colors)]
        component = np.asarray(pattern, dtype=float)
        if component.size != cumulative.size:
            fitted_component = np.zeros_like(cumulative)
            copy_count = min(component.size, cumulative.size)
            fitted_component[:copy_count] = component[:copy_count]
            component = fitted_component
        new_top = cumulative + np.maximum(component, 0)
        ax_main.fill_between(
            tt, cumulative, new_top, color=color,
            alpha=0.34 if theme == 'light' else 0.42,
            linewidth=0, zorder=1)
        ax_main.plot(tt, new_top, color=color, linewidth=0.55,
                     alpha=0.85, zorder=2)
        cumulative = new_top

    ax_main.plot(tt, y_bg, color=muted_color, linewidth=0.9,
                 linestyle=(0, (4, 2)), alpha=0.95, zorder=2)
    ax_main.plot(tt, y_obs, color=palette['observed'], linewidth=0.65,
                 alpha=0.9, zorder=3)
    ax_main.plot(tt, y_calc, color=palette['calculated'], linewidth=1.15,
                 alpha=1.0, zorder=4)

    stats_text = (
        f"$R_{{\\mathrm{{wp}}}}$ = {stats['Rwp']} %    "
        f"$R_{{\\mathrm{{p}}}}$ = {stats['Rp']} %    "
        f"$\\chi^2$ = {stats['chi2']}    "
        f"goodness of fit = {stats['GoF']}"
    )
    # Keep statistics outside the data region so the annotation can never hide
    # a high-intensity reflection.
    ax_main.set_title(
        stats_text, loc='right', pad=8, fontsize=8.3, color=text_color,
        bbox=dict(boxstyle='round,pad=0.30', fc=palette['stats_face'],
                  ec=grid_color, alpha=0.96, linewidth=0.7))

    wavelength_label = metadata.get('wavelength_label')
    if not wavelength_label:
        wavelength_label = f"λ={result.get('wavelength', 1.54056):.4f} Å"
    wavelength_label = format_wavelength_label(wavelength_label)
    method_label = clean_descriptive_text(metadata.get('method', 'Le Bail'))
    sample_label = clean_descriptive_text(metadata.get('sample_id', 'Sample'))
    fig.suptitle(sample_label, color=text_color, fontsize=12.5,
                 fontweight='bold', y=0.988)
    fig.text(
        0.5, 0.954, f"{method_label} refinement  •  {wavelength_label}",
        ha='center', va='top', color=muted_color, fontsize=9.3)

    ax_main.set_ylabel('Intensity (arbitrary units)', fontsize=10,
                       color=text_color)
    ax_main.set_ylim(bottom=0)

    legend_handles = [
        Line2D([0], [0], color=palette['observed'], lw=1.3,
               label='Observed intensity'),
        Line2D([0], [0], color=palette['calculated'], lw=1.6,
               label='Calculated total pattern'),
        Line2D([0], [0], color=muted_color, lw=1.1,
               ls=(0, (4, 2)), label='Fitted background'),
    ]
    for index, phase in enumerate(phases):
        color = phase_colors[index % len(phase_colors)]
        legend_handles.append(Patch(
            facecolor=color, edgecolor=color, alpha=0.55,
            label=phase_legend_label(phase, index=index)))
    figure_legend = fig.legend(
        handles=legend_handles, fontsize=8.1, ncol=legend_columns,
        facecolor=palette['stats_face'], edgecolor=grid_color,
        labelcolor=text_color, loc='upper center',
        bbox_to_anchor=(0.5, 0.925), frameon=True, fancybox=False,
        borderpad=0.55, columnspacing=1.0, handlelength=1.7,
        handletextpad=0.55,
    )
    figure_legend.get_frame().set_linewidth(0.7)

    # One spacious row per phase.  Vertical hkl labels show every reflection
    # without running into neighboring labels at manuscript width.
    ax_ticks.set_yticks([])
    ax_ticks.yaxis.set_visible(False)
    ax_ticks.set_ylim(0, max(n_phases, 1))
    for index, phase in enumerate(phases):
        color = phase_colors[index % len(phase_colors)]
        row_center = max(n_phases, 1) - index - 0.5
        ax_ticks.hlines(row_center - 0.23, tt.min(), tt.max(),
                        color=grid_color, linewidth=0.5, alpha=0.6)
        labeled_positions = set()
        for reflection in phase.get('tick_reflections', []) or []:
            position = float(reflection['two_theta'])
            label = reflection.get('label') or ''
            ax_ticks.vlines(position, row_center - 0.20, row_center + 0.02,
                            color=color, linewidth=1.0, alpha=0.95)
            if label:
                ax_ticks.text(
                    position, row_center + 0.06, label,
                    ha='center', va='bottom', rotation=90,
                    rotation_mode='anchor', fontsize=7.3,
                    color=text_color, clip_on=True)
            labeled_positions.add(round(position, 3))
        for position in phase.get('tick_positions', []) or []:
            if round(float(position), 3) not in labeled_positions:
                ax_ticks.vlines(float(position), row_center - 0.20,
                                row_center + 0.02, color=color,
                                linewidth=1.0, alpha=0.95)
        ax_ticks.text(
            0.006, row_center / max(n_phases, 1), f"Phase {index + 1}",
            transform=ax_ticks.transAxes, ha='left', va='center',
            fontsize=7.8, color=color, fontweight='bold',
            bbox=dict(boxstyle='square,pad=0.18', fc=surface,
                      ec='none', alpha=0.9))
    ax_ticks.set_ylabel('Reflection\npositions', fontsize=9,
                        color=text_color, labelpad=11)

    ax_res.plot(tt, resid, color=palette['residual'], linewidth=0.8,
                alpha=0.95)
    ax_res.axhline(0, color=palette['zero'], linewidth=0.75,
                   linestyle=(0, (4, 2)), alpha=0.85)
    ax_res.fill_between(tt, resid, 0, where=(resid > 0),
                        color=palette['residual'], alpha=0.16)
    ax_res.fill_between(tt, resid, 0, where=(resid < 0),
                        color=palette['calculated'], alpha=0.16)
    ax_res.set_ylabel('Observed −\ncalculated', fontsize=9, color=text_color)
    ax_res.set_xlabel('Diffraction angle, 2θ (degrees)', fontsize=10,
                      color=text_color)

    ax_main.set_xlim(tt.min(), tt.max())
    top_margin_inches = 0.72 + 0.27 * legend_rows
    fig.subplots_adjust(
        left=0.14, right=0.985, bottom=0.085,
        top=1.0 - top_margin_inches / figure_height)

    plt.savefig(
        output_path, dpi=300, bbox_inches='tight', pad_inches=0.08,
        facecolor=background,
        metadata={
            'Title': f"{sample_label} XRD refinement",
            'Description': (
                f"{method_label} refinement; {wavelength_label}; {theme} theme; "
                "phase contributions shown as shaded fitted components"
            ),
            'Software': 'Catalysis Data Toolkit',
        },
    )
    plt.close(fig)
    return output_path


def make_candidate_preview(tt, y_obs, candidates, wavelength, output_path):
    """Render the pre-refinement candidate stick-pattern preview."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#e6edf3', labelsize=8)
    ax.set_xlabel('Diffraction angle, 2θ (degrees)', fontsize=9,
                  color='#e6edf3')
    ax.set_ylabel('Intensity (arbitrary units)', fontsize=9, color='#e6edf3')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d333b')
    ax.grid(True, color='#2d333b', alpha=0.4, linewidth=0.5)

    ax.plot(tt, y_obs, color='#58a6ff', linewidth=0.8, alpha=0.9,
            label='Observed intensity', zorder=3)

    for index, candidate in enumerate(candidates[:6]):
        color = PHASE_COLORS[index % len(PHASE_COLORS)]
        for stick in candidate.get('stick_pattern', []):
            ax.axvline(stick['two_theta'], color=color, linewidth=0.8,
                       alpha=0.5, linestyle='--', ymin=0, ymax=0.15)
        formula = format_chemical_formula(candidate.get('formula', '?'))
        ax.text(
            0.01 + index * 0.16, 0.97, f"●  {formula}",
            transform=ax.transAxes, ha='left', va='top', fontsize=7,
            color=color,
            bbox=dict(boxstyle='round,pad=0.2', fc='#1c2128',
                      ec=color, alpha=0.8))

    ax.set_title('Phase identification candidate overlay', color='#e6edf3',
                 fontsize=10, fontweight='bold')
    ax.set_xlim(np.min(tt), np.max(tt))
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close(fig)
    return output_path
