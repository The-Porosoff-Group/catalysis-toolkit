import os
import sys
import tempfile
import unittest
import zipfile
import json
from unittest import mock

import numpy as np
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
    _normalize_injection_species,
    _reaction_mask,
    _select_bypass_data,
    build_flow_table,
    load_reaction_config,
    make_plots,
    validate_bypass_settings,
)
from modules.json_safety import json_safe_value  # noqa: E402


class GcRegressionTests(unittest.TestCase):
    def test_argon_o2_header_alias_supports_flow_calculation(self):
        config = load_reaction_config(os.path.join(
            ROOT, 'modules', 'reaction_configs', 'rwgs.yaml'))
        data = {
            'injections': [{
                'label': 'sample Rxn 1',
                'inj_num': 1,
                'is_bypass': False,
                'amounts': {
                    'Argon/O2': 15.0,
                    'Carbon Dioxide': 10.0,
                },
                'areas': {'Argon/O2': 150.0},
                'source_refs': {
                    'amounts': {'Argon/O2': 'K6'},
                    'areas': {'Argon/O2': 'L6'},
                },
            }],
        }

        _normalize_injection_species(data, config['species'])
        injection = data['injections'][0]
        self.assertEqual(injection['amounts']['Ar/O2'], 15.0)
        self.assertNotIn('Argon/O2', injection['amounts'])
        self.assertEqual(injection['source_refs']['amounts']['Ar/O2'], 'K6')
        self.assertIn(
            {'raw': 'Argon/O2', 'canonical': 'Ar/O2'},
            data['species_aliases_applied'])

        flows, _ = build_flow_table(data, 15.0, config['species'])
        self.assertEqual(flows.loc[0, 'Ar'], 15.0)
        self.assertEqual(flows.loc[0, 'CO2'], 10.0)

    def test_non_finite_values_are_converted_to_json_null(self):
        safe = json_safe_value({
            'balance': float('nan'),
            'nested': [np.inf, np.float64(-np.inf), np.array([1, np.nan])],
            'conversion': 4.25,
        })

        self.assertIsNone(safe['balance'])
        self.assertEqual(safe['nested'], [None, None, [1.0, None]])
        encoded = json.dumps(safe, allow_nan=False)
        self.assertNotIn('NaN', encoded)
        self.assertNotIn('Infinity', encoded)

    def test_bypass_omit_cannot_exceed_points_used(self):
        invalid = {
            'bypass_omit_initial': 4,
            'bypass_points_used': 3,
        }
        with self.assertRaisesRegex(
                ValueError, 'must be less than or equal'):
            validate_bypass_settings(invalid)
        with self.assertRaisesRegex(
                ValueError, 'must be less than or equal'):
            _select_bypass_data(
                {'injections': [{'label': f'Bypass {i}'} for i in range(6)]},
                invalid)

        valid = {
            'bypass_omit_initial': 3,
            'bypass_points_used': 3,
        }
        selected = _select_bypass_data(
            {'injections': [{'label': f'Bypass {i}'} for i in range(6)]},
            valid)
        self.assertEqual(selected['bypass_selected_points'], 3)
        self.assertEqual(
            [row['label'] for row in selected['injections']],
            ['Bypass 3', 'Bypass 4', 'Bypass 5'])

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

    def test_labeled_leak_checks_follow_blank_exclusion_path(self):
        frame = pd.DataFrame([
            {'label': 'Leak Check 1', 'inj_num': 1, 'is_bypass': False},
            {'label': 'leak-check 2', 'inj_num': 2, 'is_bypass': False},
            {'label': 'LeakCheck 3', 'inj_num': 3, 'is_bypass': False},
            {'label': 'sample bypass 1', 'inj_num': 4, 'is_bypass': True},
            {'label': 'sample Rxn 14', 'inj_num': 14, 'is_bypass': False},
            {'label': 'sample Rxn 15', 'inj_num': 15, 'is_bypass': False},
        ])
        metadata = {'injection_interval_min': 22}
        result, count = _add_time_on_stream_column(frame, metadata)

        self.assertEqual(count, 2)
        self.assertEqual(metadata['blank_excluded_points'], 3)
        self.assertListEqual(
            result.loc[:2, 'is_blank'].tolist(), [True, True, True])
        self.assertListEqual(
            result.loc[:2, 'analysis_include'].tolist(), [False, False, False])
        self.assertTrue(result.loc[:2, 'time_on_stream_h'].isna().all())
        self.assertTrue(
            (result.loc[:2, 'row_status'] == 'Blank / excluded').all())
        self.assertEqual(int(_reaction_mask(result).sum()), 2)
        self.assertEqual(result.loc[4, 'time_on_stream_h'], 0.0)
        self.assertAlmostEqual(result.loc[5, 'time_on_stream_h'], 22 / 60)

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
