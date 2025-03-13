try:
    from .cvmm import CVMMSel, cvmm, cvmm_prepare_sel2
    from .router import ApproximateTopkRouter
except Exception as e:
    print(f"Could not import CVMM: {e}")
