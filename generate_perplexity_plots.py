import matplotlib.pyplot as plt
import numpy as np

perp_vs_d = [
    (0.1, 366360),
    (0.2, 327327),
    (0.3, 308762),
    (0.4, 298142),
    (0.5, 291996),
]

d, perplexity = zip(*perp_vs_d)
plt.scatter(d, perplexity)
plt.plot(d, perplexity)
plt.title('Perplexity vs. D for Absolute Discounting')
plt.show()

perp_vs_k = [
    (1, 43156208),
    (2, 68445717),
    (3, 88325944),
    (4, 105059050),
    (5, 119654173),
]

k, perplexity = zip(*perp_vs_k)
plt.scatter(k, perplexity)
plt.plot(k, perplexity)
plt.ticklabel_format(style='plain')
plt.title('Perplexity vs. k for Laplace Smoothing')
plt.show()