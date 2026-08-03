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
    _reaction_mask,
)


class GcRegressionTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
