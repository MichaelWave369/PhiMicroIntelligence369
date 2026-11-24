import unittest
from topology_partition import partitions,choose,split_rows,score
from dataset import build,NINE_DOMAINS
class TestPartition(unittest.TestCase):
    def test_enumeration(self):
        gs=list(partitions(NINE_DOMAINS))
        self.assertEqual(len(gs),280)
        self.assertEqual(len(set(gs)),280)
        for g in gs:self.assertEqual(set(sum(g,())),set(NINE_DOMAINS))
    def test_affinity(self):
        affinity={a:{b:float(i//3==j//3) for j,b in enumerate(NINE_DOMAINS)} for i,a in enumerate(NINE_DOMAINS)}
        g,n=choose(NINE_DOMAINS,affinity)
        self.assertEqual(n,280);self.assertEqual(score(g,affinity),9)
        self.assertEqual(g,tuple(tuple(NINE_DOMAINS[i:i+3]) for i in range(0,9,3)))
    def test_split(self):
        train,test=build("nine");a,b=split_rows(train)
        self.assertEqual((len(a),len(b),len(test)),(108,54,99))
        self.assertEqual(len({r["question"] for r in a+b+test}),261)
        self.assertTrue(all(r["split"]=="calibration" for r in b))
if __name__=="__main__":unittest.main()
