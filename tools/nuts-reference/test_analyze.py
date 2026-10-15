"""Analyzer contracts, executed with the pinned ArviZ reference environment."""
import json
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from analyze import load,serializable,summarize

class Contracts(unittest.TestCase):
    def test_nonfinite_artifact_fields_are_preserved_as_null(self):
        result=serializable(dict(passed=False,values=[float('nan'),float('inf'),1.0]))
        self.assertEqual(result,dict(passed=False,values=[None,None,1.0]))
        json.dumps(result,allow_nan=False)

    def test_incomplete_or_nonfinite_draws_fail_before_diagnostics(self):
        for data in (np.zeros((3,8)),np.full((4*4*8192,8),np.nan)):
            with patch('analyze.np.loadtxt',return_value=data):
                with self.assertRaisesRegex(ValueError,'shape or values'):load(Path('unused'))

    def test_duplicate_chain_identifiers_fail(self):
        with patch('analyze.np.loadtxt',return_value=np.zeros((4*4*8192,8))):
            with self.assertRaisesRegex(ValueError,'reordered'):load(Path('unused'))

    def test_stuck_chains_fail_instead_of_being_omitted(self):
        with np.errstate(all='ignore'):
            result=summarize(np.zeros((4,4,8192,8)),'synthetic')
        self.assertEqual(len(result),20)
        self.assertTrue(all(not r['passed'] for r in result))
        json.dumps(serializable(result),allow_nan=False)

if __name__=='__main__':unittest.main()
