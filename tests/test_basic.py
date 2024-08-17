from cobweb.cobweb_discrete import CobwebTree

def test_loading():
    x = CobwebTree()
    x.ifit({1: {1: 2.0}}, 0)

