try:
    from .cvmm import CVMMSel, cvmm, cvmm_prepare_sel2
except RuntimeError as e:
    print(f"Could not import CVMM: {e}")
from .router import ApproximateTopkRouter
