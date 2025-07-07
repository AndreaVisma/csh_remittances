
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import zero_values_before_first_positive_and_after_first_negative

shape = 0.52
height = 0.215

xs = np.linspace(0, 12, 100)
ys = zero_values_before_first_positive_and_after_first_negative([height + shape * np.sin((np.pi / 6) * x) for x in xs])
y = [y for y in ys if y !=0]

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(xs[:len(y)], y, linewidth = 2)
plt.grid()
fig.savefig('.\plots\\for_paper\\disasters_function.png')
plt.show(block = True)