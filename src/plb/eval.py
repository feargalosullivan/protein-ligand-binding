"""CASF-2016 evaluation metrics (scoring + ranking power).

Implementation lands in Phase 4. The "scoring power" is global Pearson R,
Spearman R, RMSE, MAE; the "ranking power" is the mean Spearman R within each
of the 57 protein clusters that CASF-2016 defines.
"""

from __future__ import annotations
