import json,tempfile,unittest
from pathlib import Path
from report import aggregate

class Aggregation(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def shard(self,i=0,n=1,processed=2,fail=False):
        path=self.root/str(i);path.mkdir()
        row=dict(shard=i,shards=n,profile='smoke',global_cases=2,assigned=2//n+int(i<2%n),processed=processed,complete=processed==2//n+int(i<2%n),counts=dict(density_finite=processed,fused_finite=processed-int(fail),fused_error=int(fail)))
        (path/'summary.json').write_text(json.dumps(row));(path/'failures.jsonl').write_text(json.dumps(dict(index=i,kind='fused'))+'\n' if fail else '')
        return path/'summary.json'
    def test_complete_success(self):self.assertEqual(aggregate([self.shard()])['verdict'],'finite-output pass')
    def test_missing_shard_is_incomplete(self):self.assertFalse(aggregate([self.shard(n=2,processed=1)])['complete'])
    def test_retained_failure_counts(self):self.assertEqual(aggregate([self.shard(fail=True)])['verdict'],'finite-output failures')
    def test_duplicate_shard_rejected(self):
        path=self.shard()
        with self.assertRaises(ValueError):aggregate([path,path])
    def test_missing_failure_rejected(self):
        path=self.shard(fail=True);path.with_name('failures.jsonl').write_text('')
        with self.assertRaises(ValueError):aggregate([path])
    def test_forged_completeness_rejected(self):
        path=self.shard(processed=1);row=json.loads(path.read_text());row['complete']=True;path.write_text(json.dumps(row))
        with self.assertRaises(ValueError):aggregate([path])
if __name__=='__main__':unittest.main()
