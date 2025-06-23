"""
Day-dependent MSGARCH research package
"""
from .utils      import EPS, S, HUBER_C, TEMP0, RIDGE_TAU, scad_clip
from .params     import MSGARCHParams, initialize_parameters
from .em_core    import (forward_backward_EM, M_step,
                         em_fit_ms_garch, fit_ms_garch_multi)
from .simulator  import simulate_ms_garch