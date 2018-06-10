import sys
sys.path.append("../")
from magmod import sg
import numpy as np
from magbias_experiments import LSST


#first for LSST:
name = "/Users/amadeus/Documents/PhD/work/MagBias/sg_data/sg_LSST.txt"
LSST_now = dict(LSST)
if "sg_file" in LSST_now.keys(): del LSST_now["sg_file"] #otherwise sg will load the previous fit
N = 200
ztab = np.linspace(0,4,N)
sgtab = sg(ztab, LSST_now)

tab = np.array([ztab, sgtab]).T
np.savetxt(name, tab, header = "z \t sg, where sg is the magnification bias")

LSST_now.update({"sg_file":name})
print LSST_now
