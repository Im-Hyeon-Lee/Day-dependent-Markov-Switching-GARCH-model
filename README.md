# Day-dependent-Markov-switching-GARCH-model

Day-dependent MSGARCH model from Lee (2025, Calendar-based clustering of weekly extremes: Empirical failure of stochastic models)

### Modules

- `utils.py`: constants and helpers (EPS, S, scad_clip, Huber weight…)
- `params.py`: MSGARCHParams class and initialise_parameters (K-means μ, weekday transition matrix, GARCH seeds)
- `em_core.py`: EM implementation: forward_backward_EM, M_step, em_fit_ms_garch, fit_ms_garch_multi
- `simulator.py`: simulate_ms_garch for generating synthetic return / price paths

### Installation

1. Clone the repository.
2. Change into the project directory.
3. Run: `pip install -r requirements.txt`
   (All runtime dependencies are listed in `requirements.txt`)

### Usage example

```python
import numpy as np
from utils import compute_lam_scad
from em_core import em_fit_ms_garch
from simulator import simulate_ms_garch

ret = np.random.standard_normal(1000)
dows = np.arange(1000) % 5
lam = compute_lam_scad(ret)

par = em_fit_ms_garch(ret, dows, lam_scad=lam, K=2)
prices, _, _, _ = simulate_ms_garch(par, dows, lam_scad=lam, T=len(ret), P0=10_000)
