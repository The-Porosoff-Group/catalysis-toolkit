import os
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

import pandas as pd
from openpyxl import Workbook


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.gc_processor import (  # noqa: E402
    _add_time_on_stream_column,
    _copy_source_sheet_to_workbook,
    _draw_legend_label,
    _metadata_bool,
    _reaction_mask,
    load_reaction_config,
    make_plots,
)


class GcRegressionTests(unittest.TestCase):
    def test_plot_setting_boolean_values_are_normalized(self):
        for value in (True, 1, 'true', 'yes', 'on', 'checked'):
            with self.subTest(value=value):
                self.assertTrue(
                    _metadata_bool({'show_carbon_balance': value},
                                   'show_carbon_balance'))
        for value in (False, 0, 'false', 'no', 'off', 'unchecked', ''):
            with self.subTest(value=value):
                self.assertFalse(
                    _metadata_bool({'show_carbon_balance': value},
                                   'show_carbon_balance', True))

    def test_chemical_formula_digits_are_drawn_as_subscripts(self):
        class Font:
            def __init__(self, size):
                self.size = size

        class Draw:
            def __init__(self):
                self.calls = []

            def textbbox(self, _position, _text, font=None):
                return (0, 0, font.size, font.size)

            def text(self, position, text, fill=None, font=None):
                self.calls.append((position, text, font.size))

        draw = Draw()
        _draw_legend_label(
            draw, 0, 0, 'CO2 Conversion', Font(28), Font(18))
        oxygen = next(call for call in draw.calls if call[1] == 'O')
        subscript = next(call for call in draw.calls if call[1] == '2')
        self.assertGreater(subscript[0][1], oxygen[0][1])
        self.assertEqual(subscript[2], 18)

    def test_plot_renderer_does_not_change_with_timing_metadata(self):
        standard_path = os.path.join(ROOT, 'standard.png')
        cases = (
            {},
            {'run_duration_h': 12, 'injection_interval_min': 22},
            {'plot_style': 'single_axis_stacked'},
        )
        with mock.patch(
                'modules.gc_processor._draw_gc_plot',
                return_value=standard_path) as standard_renderer:
            for metadata in cases:
                with self.subTest(metadata=metadata):
                    result = make_plots(
                        None, None, None, 0, 'CO2', None, metadata,
                        None, None, ROOT)
                    self.assertEqual(result, standard_path)
        self.assertEqual(standard_renderer.call_count, len(cases))

    def test_co2_reaction_hydrogen_defaults_are_30_sccm(self):
        config_dir = os.path.join(ROOT, 'modules', 'reaction_configs')
        for filename in ('co2_hydrogenation.yaml', 'rwgs.yaml'):
            config = load_reaction_config(os.path.join(config_dir, filename))
            defaults = {
                item['label']: item['default_sccm']
                for item in config['inlet_species']
            }
            self.assertEqual(defaults['H2'], 30, filename)

    def _minimal_gc_xlsx(self, path):
        main_ns = (
            'http://schemas.openxmlformats.org/spreadsheetml/2006/main')
        rel_ns = (
            'http://schemas.openxmlformats.org/officeDocument/2006/relationships')
        pkg_ns = (
            'http://schemas.openxmlformats.org/package/2006/relationships')
        workbook = f'''<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="{main_ns}" xmlns:r="{rel_ns}">
  <sheets><sheet name="GC" sheetId="1" r:id="rId1"/></sheets>
</workbook>'''
        rels = f'''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="{pkg_ns}">
  <Relationship Id="rId1" Type="{rel_ns}/worksheet"
                Target="worksheets/sheet1.xml"/>
</Relationships>'''
        sheet = f'''<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="{main_ns}"><sheetData>
  <row r="6">
    <c r="A6" t="inlineStr"><is><t>sample Rxn 1</t></is></c>
    <c r="I6"><v>27.2</v></c>
    <c r="K6"><v>15</v></c>
  </row>
</sheetData></worksheet>'''
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('xl/workbook.xml', workbook)
            archive.writestr('xl/_rels/workbook.xml.rels', rels)
            archive.writestr('xl/worksheets/sheet1.xml', sheet)

    def test_raw_sheet_copy_falls_back_when_styles_cannot_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, 'source.xlsx')
            self._minimal_gc_xlsx(source)
            output = Workbook()
            output.active.title = 'Processed'
            with mock.patch(
                    'openpyxl.load_workbook',
                    side_effect=TypeError('malformed source styles')):
                raw = _copy_source_sheet_to_workbook(
                    output, source, worksheet_index=0,
                    sheet_name='Raw Original', insert_at=1)

            self.assertEqual(raw['A6'].value, 'sample Rxn 1')
            self.assertAlmostEqual(raw['I6'].value, 27.2)
            self.assertEqual(raw['K6'].value, 15)

    def test_labeled_blanks_are_preserved_but_excluded(self):
        frame = pd.DataFrame([
            {'label': 'sample blank 1', 'inj_num': 1, 'is_bypass': False},
            {'label': 'sample bypass 1', 'inj_num': 2, 'is_bypass': True},
            {'label': 'sample Rxn 12', 'inj_num': 12, 'is_bypass': False},
            {'label': 'sample Rxn 13', 'inj_num': 13, 'is_bypass': False},
        ])
        metadata = {'injection_interval_min': 22}
        result, count = _add_time_on_stream_column(frame, metadata)

        self.assertEqual(count, 2)
        self.assertEqual(metadata['blank_excluded_points'], 1)
        self.assertTrue(result.loc[0, 'is_blank'])
        self.assertFalse(result.loc[0, 'analysis_include'])
        self.assertEqual(result.loc[0, 'row_status'], 'Blank / excluded')
        self.assertTrue(pd.isna(result.loc[0, 'time_on_stream_h']))
        self.assertEqual(int(_reaction_mask(result).sum()), 2)
        self.assertEqual(result.loc[2, 'time_on_stream_h'], 0.0)
        self.assertAlmostEqual(result.loc[3, 'time_on_stream_h'], 22 / 60)

    def test_final_reaction_points_are_preserved_but_excluded(self):
        frame = pd.DataFrame([
            {'label': 'sample Rxn 1', 'inj_num': 1, 'is_bypass': False},
            {'label': 'sample Rxn 2', 'inj_num': 2, 'is_bypass': False},
            {'label': 'sample Rxn 3', 'inj_num': 3, 'is_bypass': False},
            {'label': 'sample Rxn 4', 'inj_num': 4, 'is_bypass': False},
        ])
        metadata = {
            'injection_interval_min': 30,
            'rejected_final_injections': 2,
        }
        result, count = _add_time_on_stream_column(frame, metadata)

        self.assertEqual(count, 2)
        self.assertListEqual(
            result['analysis_include'].tolist(), [True, True, False, False])
        self.assertEqual(result.loc[2, 'row_status'],
                         'Excluded: final reaction point')
        self.assertEqual(result.loc[3, 'row_status'],
                         'Excluded: final reaction point')
        self.assertTrue(pd.isna(result.loc[2, 'time_on_stream_h']))
        self.assertTrue(pd.isna(result.loc[3, 'time_on_stream_h']))
        self.assertEqual(result.loc[0, 'time_on_stream_h'], 0.0)
        self.assertEqual(result.loc[1, 'time_on_stream_h'], 0.5)


if __name__ == '__main__':
    unittest.main()
