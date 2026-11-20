import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src import paths
from src.datasets.tabular import (
    bytes_features_to_string, dataframes_to_dataset, dataframes_to_loader,
    get_dataset_config, scale_data,
)
from src.datasets.utils import get_dataset


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class OutputPathsTests(unittest.TestCase):
    def test_outputs_are_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'output'
            elsewhere = root / 'elsewhere'
            elsewhere.mkdir()
            with patch.object(paths, 'OUTPUT_ROOT', output), working_directory(elsewhere):
                csv = paths.output_path('nv', 'example.csv')
                pd.DataFrame({'value': [1, 2]}).to_csv(csv, index=False)
                array = paths.output_path('port_selection', 'example.npy')
                np.save(array, np.array([1, 2]))
                figure = paths.figure_path('portfolio', 'example.pdf')
                self.assertEqual(csv, output / 'nv/example.csv')
                self.assertEqual(figure, output / 'figures/portfolio/example.pdf')
                self.assertTrue(figure.parent.is_dir())
                self.assertEqual(pd.read_csv(csv)['value'].tolist(), [1, 2])
                np.testing.assert_array_equal(np.load(array), [1, 2])
                self.assertFalse((elsewhere / 'output').exists())

    def test_notebook_bootstrap_from_root_and_case_study(self):
        notebooks = sorted(paths.REPO_ROOT.rglob('*.ipynb'))
        notebooks = [p for p in notebooks if '.ipynb_checkpoints' not in p.parts]
        for notebook in notebooks:
            first = next(c for c in json.loads(notebook.read_text())['cells']
                         if c['cell_type'] == 'code')
            nodes = []
            for node in ast.parse(''.join(first['source'])).body:
                if isinstance(node, ast.Import) and any(a.name != 'sys' for a in node.names):
                    break
                if isinstance(node, ast.ImportFrom) and node.module not in ('pathlib', 'src.paths'):
                    break
                nodes.append(node)
            for cwd in (paths.REPO_ROOT, paths.REPO_ROOT / 'case_study'):
                with self.subTest(notebook=notebook.name, cwd=cwd), working_directory(cwd):
                    namespace = {}
                    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(notebook), 'exec'), namespace)
                    self.assertEqual(namespace['_repo_root'], paths.REPO_ROOT)


class TabularTests(unittest.TestCase):
    def test_float_tensors_and_sensitive_attributes(self):
        X = pd.DataFrame({'feature': [1., 2.], 'sens': [0, 1]})
        y = pd.Series([3., 4.])
        plain = dataframes_to_dataset(X, y)
        grouped = dataframes_to_dataset(X, y, use_sens=True, sensitive_attributes=['sens'])
        self.assertEqual(tuple(plain.tensors[0].shape), (2, 2))
        self.assertEqual(tuple(plain.tensors[1].shape), (2,))
        self.assertEqual(tuple(grouped.tensors[2].shape), (2, 1))
        self.assertTrue(all(t.dtype == torch.float32 for t in grouped.tensors))
        torch.testing.assert_close(grouped.tensors[2], torch.tensor([[0.], [1.]]))
        X.iloc[0, 0] = 99.
        self.assertEqual(plain.tensors[0][0, 0].item(), 1.)

    def test_loader_batches(self):
        X = pd.DataFrame({'feature': [1., 2., 3.]})
        y = pd.Series([4., 5., 6.])
        batches = list(dataframes_to_loader(X, y, batch_size=2, shuffle=False))
        self.assertEqual([len(features) for features, _ in batches], [2, 1])
        torch.testing.assert_close(batches[0][1], torch.tensor([4., 5.]))

    def test_scaling_preserves_indices_column_order_and_indicator(self):
        train = pd.DataFrame({'b': [1., 3.], 'a': [10., 20.], 'sens': [0, 1]}, index=[10, 20])
        test = pd.DataFrame({'b': [5.], 'a': [30.], 'sens': [1]}, index=[30])
        validation = test.copy()
        scaled, held_out, _ = scale_data(train, test, validation, cols_to_preserve=['sens'])
        self.assertEqual(scaled.columns.tolist(), ['b', 'a', 'sens'])
        self.assertEqual(scaled.index.tolist(), [10, 20])
        self.assertEqual(held_out.index.tolist(), [30])
        np.testing.assert_allclose(scaled[['b', 'a']], [[-1., -1.], [1., 1.]])
        np.testing.assert_allclose(held_out[['b', 'a']], [[3., 3.]])
        self.assertEqual(scaled['sens'].tolist(), [0, 1])
        self.assertFalse(held_out.isna().any().any())

    def test_scaling_when_all_columns_are_preserved(self):
        frame = pd.DataFrame({'sens': [0, 1]}, index=[10, 20])
        result, _, _ = scale_data(frame, frame, frame, cols_to_preserve=['sens'])
        pd.testing.assert_frame_equal(result, frame)

    def test_byte_conversion_preserves_strings_and_missing_values(self):
        frame = pd.DataFrame({'text': [b'yes', 'no', None]})
        bytes_features_to_string(frame)
        self.assertEqual(frame['text'].tolist(), ['yes', 'no', None])

    def test_wine_default_root_and_public_loaders_from_another_directory(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(temporary):
            config = get_dataset_config('wine', random_state=0, train_frac=.3)
            self.assertEqual(Path(config.root_dir), paths.DATA_ROOT)
            with contextlib.redirect_stdout(io.StringIO()):
                wine = get_dataset(config)
            self.assertEqual(wine.X_tr.shape[1], 12)
            for split in ('train', 'test', 'validation'):
                for use_sens in (False, True):
                    batch = next(iter(wine.get_dataloader(
                        split, batch_size=4, shuffle=False, use_sens=use_sens)))
                    self.assertEqual(len(batch), 3 if use_sens else 2)
                    self.assertEqual(tuple(batch[0].shape), (4, 12))
                    self.assertTrue(all(t.dtype == torch.float32 for t in batch))

    def test_unused_broken_adapters_are_not_advertised(self):
        for name in ('compas', 'brfss'):
            with self.assertRaises(NotImplementedError):
                get_dataset_config(name)


if __name__ == '__main__':
    unittest.main()
