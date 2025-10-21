
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import zero_values_before_first_positive_and_after_first_negative

params = [2.5688294072397375, -9.9258425978672, 8.637134501037425,
            0.2337965838718275, 0.272087461937716,-0.7457719761289481,
            0.21375934475668515, 0.13041116247195136]
param_nta, param_asy, param_gdp, height, shape, shift, constant, rem_pct = params

xs = np.linspace(0, 12, 100)
ys = zero_values_before_first_positive_and_after_first_negative([height + shape * np.sin((np.pi / 6) * (x+shift)) for x in xs])
y = [y for y in ys if y !=0]

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(xs[:len(y)], y, linewidth = 2)
plt.grid()
fig.savefig('.\plots\\for_paper\\disasters_function.png')
plt.show(block = True)