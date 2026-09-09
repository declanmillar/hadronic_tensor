# Reference data shipped with the package

Small classical references the analysis needs, kept inside the package so a
bundle is self-contained.  Regenerate from the repository, not by hand.

| file | what | made by |
|---|---|---|
| `wing_surrogate_prod.npz` | staggered vacuum breathing at (m0, g2, eta) = (0.7, 1.1, 1.3): the ideal wing-anchor target per parity and time | `scripts/wing_surrogate.py build` |
| `wing_surrogate_relA.npz` | the same at (0.4, 1.4, 2.3), which is what the relA cards' deep slices anchor against | `scripts/wing_surrogate.py build` |
| `qpdf_card_refs.npz` | ideal h(m), q(x), norm and <x> for the six width cards, plus the sigma_k^2 fits | `scripts/qpdf_card_refs.py` |

Each surrogate carries its couplings and is refused for a card with a
different eta: the breathing is coupling specific, and serving a relA card
from the production surrogate would bias every anchored slice.
