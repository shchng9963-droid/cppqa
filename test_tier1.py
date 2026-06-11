import os
import sys
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.problems import load_problem


class Tier1ExperimentInterfaceTest(unittest.TestCase):
    def setUp(self):
        self.problem = load_problem('nrp_small_v2', data_dir=os.path.join(ROOT, 'benchmarks'))

    def test_default_methods_include_smsemoa(self):
        from experiment_common import DEFAULT_METHODS

        self.assertIn('SMS-EMOA', DEFAULT_METHODS)

    def test_run_method_dispatches_smsemoa(self):
        from experiment_common import budget_for, run_method

        budget = budget_for('nrp_small_v2')
        xs, fs = run_method('SMS-EMOA', self.problem, seed=0, budget=budget)

        self.assertGreater(len(xs), 0)
        self.assertEqual(len(xs), len(fs))
        for x in xs:
            _, viol = self.problem.evaluate(x)
            self.assertEqual(viol, 0)

    def test_rq5_tier1_overrides_match_curator_variants(self):
        from experiment_common import budget_for
        from run_curator_capacity_tier1 import VARIANTS as CURATOR_VARIANTS
        from run_rq5_ablation_tier1 import variant_overrides

        budget = budget_for('nrp_large_v2')
        full = variant_overrides('full', budget)
        no_curator = variant_overrides('no-Curator', budget)
        strict_nd = CURATOR_VARIANTS['strictND']['CP-PQA']
        curator = CURATOR_VARIANTS['eps002_cap64']['CP-PQA']

        self.assertEqual(full['n_rounds'], budget['atps_rounds'])
        self.assertEqual(full['num_reads'], budget['qa_reads'])
        self.assertEqual(full['decompose'], budget['decompose'])
        self.assertEqual(full['use_curator'], curator['use_curator'])
        self.assertEqual(full['curator_eps'], curator['curator_eps'])
        self.assertEqual(full['curator_capacity'], curator['curator_capacity'])

        self.assertEqual(no_curator['n_rounds'], budget['atps_rounds'])
        self.assertEqual(no_curator['num_reads'], budget['qa_reads'])
        self.assertEqual(no_curator['decompose'], budget['decompose'])
        self.assertEqual(no_curator['use_curator'], strict_nd['use_curator'])

    def test_rq5_tier1_rows_use_variant_labels_but_cp_pqa_method(self):
        from run_rq5_ablation_tier1 import rows_for_problem_variants

        archives = {
            'full': {0: [[1.0, 4.0], [2.0, 3.0]]},
            'no-Curator': {0: [[2.0, 4.0], [3.0, 2.0]]},
        }
        runtimes = {
            'full': {0: 1.0},
            'no-Curator': {0: 2.0},
        }
        feasrates = {
            'full': {0: 1.0},
            'no-Curator': {0: 1.0},
        }

        rows = rows_for_problem_variants('nrp_small_v2', ['full', 'no-Curator'], [0], archives, runtimes, feasrates)

        self.assertEqual(len(rows), 2)
        self.assertEqual({row['label'] for row in rows}, {'full', 'no-Curator'})
        self.assertEqual({row['method'] for row in rows}, {'CP-PQA'})

    def test_rq5_tier1_table_render_uses_expected_layout(self):
        from run_rq5_ablation_tier1 import render_table

        summary_rows = [
            {'problem': 'nrp_large_v2', 'label': 'full', 'HV_mean': 126693.4, 'size_mean': 64.0},
            {'problem': 'nrp_large_v2', 'label': 'no-FPE', 'HV_mean': 136527.2, 'size_mean': 30.6},
            {'problem': 'nrp_large_v2', 'label': 'no-ATPS', 'HV_mean': 317509.2, 'size_mean': 62.6},
            {'problem': 'nrp_large_v2', 'label': 'no-Curator', 'HV_mean': 126693.1, 'size_mean': 3.0},
            {'problem': 'nrp_large_v2', 'label': 'no-SHD', 'HV_mean': 126693.4, 'size_mean': 30.2},
        ]

        table = render_table(summary_rows)

        self.assertIn('\\caption{RQ5 pillar ablation across the ten benchmarks.}', table)
        self.assertIn('NRP-100 & 126693 & 64.0 & 136527 & 30.6 & 317509 & 62.6 & 126693 & 3.0 & 126693 & 30.2 \\\\', table)
        self.assertIn('HV uses a tier1-internal reference over the five variants', table)


if __name__ == '__main__':
    unittest.main()
