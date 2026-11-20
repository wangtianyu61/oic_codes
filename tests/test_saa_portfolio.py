import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src import paths
from src.portfolio_model import mv_portfolio


def load_script():
    script = paths.REPO_ROOT / 'case_study' / 'saa_portfolio.py'
    spec = importlib.util.spec_from_file_location('saa_portfolio', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SaaPortfolioTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_import_does_not_run_experiment_or_create_figure(self):
        figures = plt.get_fignums()
        with patch('src.portfolio_model.mv_portfolio', side_effect=AssertionError('experiment ran on import')), \
                patch('matplotlib.pyplot.savefig', side_effect=AssertionError('saved on import')):
            module = load_script()
        self.assertTrue(callable(module.main))
        self.assertEqual(plt.get_fignums(), figures)

    def test_reduced_experiment_plots_both_sample_sizes_and_renders_pdf(self):
        module = load_script()
        module.loop_num = 2
        module.outer_num = 2
        module.asset_nums = [2, 4]
        self.assertIs(module.mv_portfolio, mv_portfolio)
        pdf = io.BytesIO()
        with patch.object(module, 'figure_path', return_value=pdf) as target, \
                plt.rc_context({'savefig.format': 'pdf'}), \
                contextlib.redirect_stdout(io.StringIO()):
            module.main()
            target.assert_called_once_with('portfolio', 'test2.pdf')
            self.assertTrue(pdf.getvalue().startswith(b'%PDF-'))
            curves = [line for line in plt.gca().lines if not line.get_label().startswith('_')]
            self.assertEqual(len(curves), 6)
            self.assertEqual([line.get_linestyle() for line in curves], ['-'] * 3 + ['--'] * 3)


if __name__ == '__main__':
    unittest.main()
