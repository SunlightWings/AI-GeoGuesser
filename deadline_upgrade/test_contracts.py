import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from common import aligned_labels, cache_file, metrics, npz_write, read_embeddings, validate_submission
from evaluate import retrieval_predict

class Contracts(unittest.TestCase):
    def test_metric(self):
        r=metrics([[0,0],[50,10]],[[0,0],[50,10]])
        self.assertEqual(r['median_km'],0); self.assertEqual(r['lt_40_km'],1)

    def test_fold_alignment_and_missing_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); labels=pd.DataFrame(dict(filename=['a','b','c','d','e'],lat=[0]*5,lng=[0]*5,country=['X']*5))
            folds=labels.assign(fold=range(5)).iloc[::-1]
            labels.to_csv(root/'labels.csv',index=False); folds.to_csv(root/'folds.csv',index=False)
            result=aligned_labels(root/'labels.csv',root/'folds.csv')
            self.assertEqual(result.fold.tolist(),list(range(5)))
            folds.iloc[:-1].to_csv(root/'folds.csv',index=False)
            with self.assertRaises(ValueError): aligned_labels(root/'labels.csv',root/'folds.csv')

    def test_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_write(cache_file(tmp,'a.jpg'),filename=np.asarray('a.jpg'),embedding=np.asarray([3.,4.]))
            np.testing.assert_allclose(read_embeddings(tmp,['a.jpg']),[[.6,.8]])

    def test_submission_duplicates_nonfinite_and_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['a.jpg','b.jpg']: (root/name).touch()
            frame=pd.DataFrame(dict(filename=['a.jpg','b.jpg'],pred_lat=[0.,1.],pred_lng=[2.,3.]))
            self.assertTrue(validate_submission(frame,root,count=2))
            for bad in [frame.assign(filename=['a.jpg','a.jpg']),frame.assign(pred_lat=[np.nan,1]),frame.rename(columns={'pred_lat':'lat'})]:
                with self.assertRaises(ValueError): validate_submission(bad,root,count=2)

    def test_retrieval_gating_and_blend(self):
        bank=dict(embedding=np.eye(2),truth=np.array([[0,0],[50,10]]),true_country=np.array([0,1]))
        query=dict(embedding=np.array([[1.,0.]]),country=np.array([1]),pred=np.array([[20.,20.]]))
        cfg=dict(k=1,temperature=.1,country_gate=False,blend=1.)
        np.testing.assert_allclose(retrieval_predict(query,bank,cfg),[[0,0]],atol=1e-4)
        np.testing.assert_allclose(retrieval_predict(query,bank,{**cfg,'country_gate':True}),[[50,10]],atol=1e-4)
        np.testing.assert_allclose(retrieval_predict(query,bank,{**cfg,'blend':0.}),[[20,20]],atol=1e-4)

    def test_gated_k_larger_than_country_members(self):
        bank=dict(embedding=np.eye(2),truth=np.array([[0,0],[50,10]]),true_country=np.array([0,1]))
        query=dict(embedding=np.array([[1.,0.]]),country=np.array([1]),pred=np.array([[20.,20.]]))
        out=retrieval_predict(query,bank,dict(k=10,temperature=.1,country_gate=True,blend=1.))
        np.testing.assert_allclose(out,[[50,10]],atol=1e-4)

if __name__=='__main__': unittest.main()
