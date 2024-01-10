import numpy as np
from matplotlib import pyplot as plt

dataset_names = ["AG News",
                 "Clickbait Notclickbait",
                 "Financial Phrasebank"]

index = np.arange(3)

ann_likelihood = [0.913, 0.827, 0.9]
ann_success = [0.0575, 0.1321, 0.0302]

edc_likelihood = [0.871, 0.796, 0.852]
edc_success = [0.2111, 0.1309, 0.1235]

fig, ax = plt.subplots(1, 1)
ax.bar(index, ann_likelihood, width=0.25, label="ANN")
ax.bar(index+0.25, edc_likelihood, width=0.25, label="EDC")
ax.set_yticks(np.arange(0, 1.05, 0.1))
ax.set_ylim((0, 1.1))
ax.legend()
ax.set_xticks([0.125, 1.125, 2.125])
ax.set_xticklabels(dataset_names)
plt.ylabel("Likelihood")
plt.savefig("likelihood_using_uncertainty.png")
plt.show()

fig, ax = plt.subplots(1, 1)
ax.bar(index, ann_success, width=0.25, label="ANN")
ax.bar(index+0.25, edc_success, width=0.25, label="EDC")
ax.set_yticks(np.arange(0, 0.3, 0.05))
ax.set_ylim((0, 0.27))
ax.legend()
ax.set_xticks([0.125, 1.125, 2.125])
ax.set_xticklabels(dataset_names)
plt.ylabel("Success")
plt.savefig("success_of_uncertainty.png")
plt.show()
