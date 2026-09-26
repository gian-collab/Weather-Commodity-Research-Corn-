from weatherlab.validation.metrics import coverage,spread_skill
def test_metrics():
 assert coverage([1,2],[0,1],[2,3])==1
 assert spread_skill([1,2,3],[1,-2,3])>.99
