"""Fast Pillow publication plots for characterization modules."""

from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size, bold=False):
    candidates = (
        'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf',
        'arialbd.ttf' if bold else 'arial.ttf',
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, max(8, int(size)))
        except OSError:
            continue
    return ImageFont.load_default()


def _hex_rgb(value, default=(50, 130, 210)):
    text = str(value or '').lstrip('#')
    if len(text) == 3:
        text = ''.join(char * 2 for char in text)
    try:
        return tuple(int(text[index:index + 2], 16) for index in (0, 2, 4))
    except (ValueError, TypeError):
        return default


def _limits(values, requested_min=None, requested_max=None, include_zero=False):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    low = float(np.min(finite)) if len(finite) else 0.0
    high = float(np.max(finite)) if len(finite) else 1.0
    if include_zero:
        low, high = min(low, 0.0), max(high, 0.0)
    span = high - low
    if span <= 0:
        span = max(abs(high), 1.0)
    low -= span * 0.05
    high += span * 0.08
    return (float(requested_min) if requested_min is not None else low,
            float(requested_max) if requested_max is not None else high)


def _tick_values(low, high, count=6):
    if high <= low:
        return [low]
    return np.linspace(low, high, count)


def _format_tick(value):
    magnitude = abs(value)
    if magnitude >= 10000 or (magnitude and magnitude < 0.001):
        return f'{value:.2e}'
    if magnitude >= 100:
        return f'{value:.0f}'
    if magnitude >= 10:
        return f'{value:.1f}'
    return f'{value:.3g}'


def _draw_axes(draw, rect, xlim, ylim, x_label, y_label, fonts, show_grid=False,
               show_y_tick_labels=True):
    left, top, right, bottom = rect
    axis_color = (45, 51, 59)
    grid_color = (220, 225, 231)
    tick_font, axis_font = fonts
    for value in _tick_values(*xlim):
        x = left + (value - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)
        if show_grid:
            draw.line((x, top, x, bottom), fill=grid_color, width=1)
        draw.line((x, bottom, x, bottom + 6), fill=axis_color, width=2)
        label = _format_tick(value)
        box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (box[2] - box[0]) / 2, bottom + 9), label, font=tick_font, fill=axis_color)
    y_tick_width = 0
    for value in _tick_values(*ylim):
        y = bottom - (value - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top)
        if show_grid:
            draw.line((left, y, right, y), fill=grid_color, width=1)
        draw.line((left - 6, y, left, y), fill=axis_color, width=2)
        if show_y_tick_labels:
            label = _format_tick(value)
            box = draw.textbbox((0, 0), label, font=tick_font)
            label_width = box[2] - box[0]
            y_tick_width = max(y_tick_width, label_width)
            draw.text((left - 8 - label_width, y - (box[3] - box[1]) / 2),
                      label, font=tick_font, fill=axis_color)
    draw.line((left, bottom, right, bottom), fill=axis_color, width=2)
    draw.line((left, top, left, bottom), fill=axis_color, width=2)
    box = draw.textbbox((0, 0), x_label, font=axis_font)
    draw.text(((left + right - (box[2] - box[0])) / 2, bottom + 34),
              x_label, font=axis_font, fill=axis_color)
    label_box = axis_font.getbbox(y_label)
    label_image = Image.new('RGBA', (label_box[2] - label_box[0] + 10,
                                     label_box[3] - label_box[1] + 10), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_image)
    label_draw.text((5, 5), y_label, font=axis_font, fill=axis_color)
    rotated = label_image.rotate(90, expand=True)
    return rotated, y_tick_width


def _paste_y_label(image, label_image, rect, tick_width, scale):
    """Place a vertical axis label close to its tick values without overlap."""
    left, top, _, bottom = rect
    gap = int(8 * scale)
    tick_gap = int(8 * scale) if tick_width else 0
    x = max(int(2 * scale), int(left - tick_gap - tick_width - gap - label_image.width))
    y = int((top + bottom - label_image.height) / 2)
    image.paste(label_image, (x, y), label_image)


def _legend_width(draw, items, font, scale):
    text_width = max((draw.textbbox((0, 0), label, font=font)[2]
                      for label, _ in items), default=0)
    return int(text_width + 61 * scale)


def _mapper(rect, xlim, ylim):
    left, top, right, bottom = rect

    def map_point(x, y):
        px = left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)
        py = bottom - (y - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top)
        return int(round(px)), int(round(py))
    return map_point


def _series_points(x, y, mapper, maximum=12000):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    indices = np.flatnonzero(valid)
    if len(indices) > maximum:
        indices = indices[::max(1, len(indices) // maximum)]
    return [mapper(float(x[index]), float(y[index])) for index in indices]


def _legend(draw, items, right, top, font, scale):
    widths = [draw.textbbox((0, 0), label, font=font)[2] for label, _ in items]
    text_width = max(widths, default=0)
    line_length = 35 * scale
    start = right - text_width - line_length - 18 * scale
    y = top
    for label, color in items:
        draw.line((start, y + 7 * scale, start + line_length, y + 7 * scale),
                  fill=color, width=max(2, int(2 * scale)))
        draw.text((start + line_length + 8 * scale, y), label, font=font, fill=(45, 51, 59))
        y += 21 * scale


def render_program_plot(path, x, corrected, peaks, settings):
    dpi = int(settings['png_dpi'])
    scale = max(1.0, dpi / 100.0)
    width = max(700, int(settings['figure_width'] * dpi))
    height = max(420, int(settings['figure_height'] * dpi))
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    tick_font = _font(settings['tick_font_size'] * scale)
    axis_font = _font(settings['axis_font_size'] * scale)
    title_font = _font(settings['title_font_size'] * scale, bold=True)
    legend_font = _font(settings['legend_font_size'] * scale)
    signal_color = _hex_rgb(settings['signal_color'])
    baseline_color = _hex_rgb(settings['baseline_color'], (125, 133, 144))
    peak_color = _hex_rgb(settings['peak_color'], (214, 69, 69))
    integration_color = _hex_rgb(settings['integration_color'], (139, 198, 236))
    legend = [('Corrected signal', signal_color)]
    if settings['show_baseline']:
        legend.append(('Baseline', baseline_color))
    if settings['show_integration']:
        legend.append(('Integrated area', integration_color))
    right_gutter = max(int(42 * scale), _legend_width(draw, legend, legend_font, scale) + int(18 * scale))
    rect = (int(112 * scale), int(65 * scale), width - right_gutter, height - int(82 * scale))
    finite_x = np.asarray(x, dtype=float)
    finite_x = finite_x[np.isfinite(finite_x)]
    xlim = (float(settings['x_axis_min']) if settings.get('x_axis_min') is not None else float(np.min(finite_x)),
            float(settings['x_axis_max']) if settings.get('x_axis_max') is not None else float(np.max(finite_x)))
    if xlim[1] <= xlim[0]:
        xlim = (xlim[0], xlim[0] + 1.0)
    ylim = _limits(corrected, settings.get('y_axis_min'), settings.get('y_axis_max'), include_zero=True)
    if settings.get('show_peak_labels') and settings.get('y_axis_max') is None:
        ylim = (ylim[0], ylim[1] + max(ylim[1] - ylim[0], 1e-12) * 0.14)
    y_label, y_tick_width = _draw_axes(
        draw, rect, xlim, ylim, settings['x_axis_label'], settings['y_axis_label'],
        (tick_font, axis_font), settings['show_grid'], settings.get('show_y_tick_labels', True))
    _paste_y_label(image, y_label, rect, y_tick_width, scale)
    mapper = _mapper(rect, xlim, ylim)
    points = _series_points(x, corrected, mapper)
    if settings['show_integration'] and points:
        zero_y = mapper(xlim[0], 0)[1]
        polygon = [(points[0][0], zero_y)] + points + [(points[-1][0], zero_y)]
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        ImageDraw.Draw(overlay).polygon(polygon, fill=integration_color + (68,))
        image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(image)
    if len(points) >= 2:
        draw.line(points, fill=signal_color, width=max(2, int(settings['line_width'] * scale)), joint='curve')
    if settings['show_baseline']:
        y0 = mapper(xlim[0], 0)[1]
        dash = max(5, int(8 * scale))
        cursor = rect[0]
        while cursor < rect[2]:
            draw.line((cursor, y0, min(cursor + dash, rect[2]), y0), fill=baseline_color,
                      width=max(1, int(scale)))
            cursor += dash * 2
    if settings['show_peak_markers'] or settings['show_peak_labels']:
        label_font = _font(max(8, settings['tick_font_size'] - 1) * scale)
        label_limit = max(0, int(settings.get('max_peak_labels', 10)))
        ranked = sorted(range(len(peaks)), key=lambda index: peaks[index].get('prominence', 0), reverse=True)
        labeled = set(ranked[:label_limit])
        for position, peak in enumerate(peaks):
            if peak.get('index') is None:
                continue
            index = int(peak['index'])
            if index < 0 or index >= len(x):
                continue
            px, py = mapper(float(x[index]), float(corrected[index]))
            if not (rect[0] <= px <= rect[2] and rect[1] <= py <= rect[3]):
                continue
            if settings['show_peak_markers']:
                radius = max(3, int(4 * scale))
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=peak_color)
            if settings['show_peak_labels'] and position in labeled:
                label = (f"{peak['time_min']:.2f} min" if settings.get('x_axis_basis') == 'time'
                         else (f"{peak['temperature_c']:.1f} °C"
                               if peak.get('temperature_c') is not None
                               else f"{peak['time_min']:.2f} min"))
                box = draw.textbbox((0, 0), label, font=label_font)
                offset = int((12 + (position % 3) * 11) * scale)
                label_width = box[2] - box[0]
                label_height = box[3] - box[1]
                label_x = min(max(px - label_width / 2, rect[0] + 4 * scale),
                              rect[2] - label_width - 4 * scale)
                label_y = py - offset - label_height
                label_y = min(max(label_y, rect[1] + 3 * scale), rect[3] - label_height - 3 * scale)
                draw.text((label_x, label_y),
                          label, font=label_font, fill=peak_color)
    title_box = draw.textbbox((0, 0), settings['title'], font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, int(14 * scale)),
              settings['title'], font=title_font, fill=(25, 31, 39))
    _legend(draw, legend, width - int(16 * scale), rect[1] + int(8 * scale), legend_font, scale)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path, format='PNG', dpi=(dpi, dpi), optimize=True)


def render_bet_plot(path, adsorption_x, adsorption_q, desorption_x, desorption_q,
                    included_mask, slope, intercept, r_squared, settings):
    dpi = int(settings['png_dpi'])
    scale = max(1.0, dpi / 100.0)
    width = max(950, int(settings['figure_width'] * dpi))
    height = max(430, int(settings['figure_height'] * dpi))
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    tick_font = _font(settings['tick_font_size'] * scale)
    axis_font = _font(settings['axis_font_size'] * scale)
    title_font = _font(settings['title_font_size'] * scale, bold=True)
    legend_font = _font(settings['legend_font_size'] * scale)
    show_isotherm = bool(settings.get('show_isotherm', True))
    show_bet_plot = bool(settings.get('show_bet_plot', True))
    if not show_isotherm and not show_bet_plot:
        show_isotherm = True
    panel_names = (["isotherm"] if show_isotherm else []) + (["bet"] if show_bet_plot else [])
    gap = int(118 * scale) if len(panel_names) == 2 else 0
    margin_left = int(112 * scale)
    margin_right = int(35 * scale)
    top = int(68 * scale)
    bottom = height - int(80 * scale)
    panel_width = int((width - margin_left - margin_right - gap) / len(panel_names))
    rects = {}
    for position, name in enumerate(panel_names):
        left = margin_left + position * (panel_width + gap)
        rects[name] = (left, top, left + panel_width, bottom)
    adsorption_x = np.asarray(adsorption_x, dtype=float)
    adsorption_q = np.asarray(adsorption_q, dtype=float)
    desorption_x = np.asarray(desorption_x, dtype=float)
    desorption_q = np.asarray(desorption_q, dtype=float)
    adsorption_color = _hex_rgb(settings['adsorption_color'])
    desorption_color = _hex_rgb(settings['desorption_color'], (229, 139, 42))
    fit_color = _hex_rgb(settings['fit_color'], (214, 69, 69))
    window_color = _hex_rgb(settings['window_color'], (63, 174, 98))
    radius = max(2, int(settings['marker_size'] * scale / 2))
    if show_isotherm:
        rect = rects['isotherm']
        xlim_iso = (settings.get('x_axis_min') if settings.get('x_axis_min') is not None else 0.0,
                    settings.get('x_axis_max') if settings.get('x_axis_max') is not None else 1.0)
        all_quantity = (np.concatenate((adsorption_q, desorption_q))
                        if len(desorption_q) else adsorption_q)
        ylim_iso = _limits(all_quantity, settings.get('isotherm_y_min'),
                           settings.get('isotherm_y_max'), include_zero=True)
        y_label, tick_width = _draw_axes(
            draw, rect, xlim_iso, ylim_iso, settings['isotherm_x_label'],
            settings['isotherm_y_label'], (tick_font, axis_font), settings['show_grid'])
        _paste_y_label(image, y_label, rect, tick_width, scale)
        mapper_iso = _mapper(rect, xlim_iso, ylim_iso)
        ads_points = _series_points(adsorption_x, adsorption_q, mapper_iso)
        des_points = (_series_points(desorption_x, desorption_q, mapper_iso)
                      if settings['show_desorption'] and len(desorption_x) else [])
        if len(ads_points) >= 2:
            draw.line(ads_points, fill=adsorption_color,
                      width=max(2, int(settings['line_width'] * scale)))
        if ads_points and des_points:
            draw.line((ads_points[-1], des_points[0]), fill=desorption_color,
                      width=max(2, int(settings['line_width'] * scale)))
        if len(des_points) >= 2:
            draw.line(des_points, fill=desorption_color,
                      width=max(2, int(settings['line_width'] * scale)))
        for px, py in ads_points:
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=adsorption_color)
        for px, py in des_points:
            draw.rectangle((px - radius, py - radius, px + radius, py + radius), fill=desorption_color)
        if settings['show_fit_window']:
            for x, y in zip(adsorption_x[included_mask], adsorption_q[included_mask]):
                px, py = mapper_iso(float(x), float(y))
                rr = radius + max(2, int(2 * scale))
                draw.ellipse((px - rr, py - rr, px + rr, py + rr), outline=window_color,
                             width=max(2, int(scale)))
        legend_items = [('Adsorption', adsorption_color)]
        if des_points:
            legend_items.append(('Desorption', desorption_color))
        if settings['show_fit_window']:
            legend_items.append(('BET fit window', window_color))
        _legend(draw, legend_items, rect[2], rect[1] + int(8 * scale), legend_font, scale)

    if show_bet_plot:
        rect = rects['bet']
        with np.errstate(divide='ignore', invalid='ignore'):
            transform = adsorption_x / (adsorption_q * (1 - adsorption_x))
        bet_max_x = min(0.5, max(float(np.max(adsorption_x[included_mask])) * 1.2, 0.35))
        bet_mask = (adsorption_x <= bet_max_x) & np.isfinite(transform)
        xlim_bet = (0.0, bet_max_x)
        ylim_bet = _limits(transform[bet_mask], settings.get('bet_y_min'),
                           settings.get('bet_y_max'), include_zero=True)
        y_label, tick_width = _draw_axes(
            draw, rect, xlim_bet, ylim_bet, settings['bet_x_label'], settings['bet_y_label'],
            (tick_font, axis_font), settings['show_grid'])
        _paste_y_label(image, y_label, rect, tick_width, scale)
        mapper_bet = _mapper(rect, xlim_bet, ylim_bet)
        for index in np.flatnonzero(bet_mask):
            x, y = adsorption_x[index], transform[index]
            px, py = mapper_bet(float(x), float(y))
            color = window_color if included_mask[index] else (154, 164, 178)
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
        used_x = adsorption_x[included_mask]
        fit_points = [mapper_bet(float(x), float(slope * x + intercept))
                      for x in (float(np.min(used_x)), float(np.max(used_x)))]
        draw.line(fit_points, fill=fit_color,
                  width=max(2, int(settings['line_width'] * scale)))
        _legend(draw, [(f'Fit (R²={r_squared:.5f})', fit_color), ('Included', window_color)],
                rect[2], rect[1] + int(8 * scale), legend_font, scale)
    title_box = draw.textbbox((0, 0), settings['title'], font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, int(14 * scale)),
              settings['title'], font=title_font, fill=(25, 31, 39))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path, format='PNG', dpi=(dpi, dpi), optimize=True)
