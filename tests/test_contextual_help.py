import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from contextual_help import annotated_html, help_columns, help_editor, help_for, help_dataframe, TERMS


class ContextualHelpTests(unittest.TestCase):
    def test_specific_terms_and_metric_context(self):
        self.assertIn('16 × 16', help_for('SAM points_per_side'))
        self.assertIn('NMS', help_for('IoU/NMS'))
        self.assertIn('0,50 a 0,95', help_for('metrics.map50_95_box'))
        self.assertIn('caixas delimitadoras', help_for('metrics.map50_95_box'))
        self.assertIn('máscaras de pixels', help_for('metrics.recall_mask'))
        self.assertIn('milissegundos', help_for('speed_ms.inference'))
        self.assertIn('não é a taxa clássica', help_for('false_positive_rate_over_reviewed'))
        self.assertNotEqual(help_for('Batch'), help_for('Batch DINOv2'))
        self.assertIsNone(help_for('abc123'))

    def test_html_is_escaped_and_keyboard_help_is_available(self):
        rendered = annotated_html('<script>alert("SAM")</script> & IoU')
        self.assertNotIn('<script>', rendered)
        self.assertIn('&lt;script&gt;', rendered)
        self.assertIn('<summary aria-label="Ajuda sobre SAM"', rendered)
        self.assertIn('ⓘ', rendered)
        self.assertNotIn('ⓘ', annotated_html('sampler sampling'))

    def test_tables_preserve_data_and_editor_types(self):
        data = pd.DataFrame({'confidence': [0.8], 'false_positive': [False], 'custom': ['x']})
        original = data.copy(deep=True)
        cols = help_columns(data.columns)
        self.assertNotIn('custom', cols)
        self.assertNotIn('type_config', cols['false_positive'])
        for function, name in [(help_dataframe, 'dataframe'), (help_editor, 'data_editor')]:
            with patch('contextual_help.st.' + name, return_value=data) as render:
                self.assertIs(function(data), data)
                self.assertIs(render.call_args.args[0], data)
                self.assertIn('confidence', render.call_args.kwargs['column_config'])
        pd.testing.assert_frame_equal(data, original)

    def test_native_widget_help_and_inline_help_render(self):
        app = AppTest.from_string('''
import streamlit as st
import pandas as pd
from contextual_help import help_for, help_text, help_editor, help_dataframe, help_json
st.number_input("Epochs", value=100, help=help_for("Epochs"))
st.slider("Confidence", 0.0, 1.0, 0.25, help=help_for("Confidence"))
help_text("SAM e DINOv2 geram embeddings para HDBSCAN.", caption=True)
help_dataframe(pd.DataFrame({"metrics.map50_95_box": [0.5]}))
help_editor(pd.DataFrame({"false_positive": [False], "confidence": [0.8]}))
help_json({"false_positive_rate_over_reviewed": 0.2})
''').run()
        self.assertFalse(app.exception, app.exception)
        self.assertEqual(app.number_input[0].value, 100)
        self.assertIn('100 vezes', app.number_input[0].proto.help)
        self.assertEqual(app.slider[0].value, 0.25)
        self.assertTrue(any('ⓘ' in element.value for element in app.markdown))

    def test_all_definitions_are_reachable(self):
        for entry in TERMS.values():
            self.assertTrue(entry['text'].strip())
            for alias in entry['aliases']:
                self.assertIsNotNone(help_for(alias), alias)


if __name__ == '__main__':
    unittest.main()
