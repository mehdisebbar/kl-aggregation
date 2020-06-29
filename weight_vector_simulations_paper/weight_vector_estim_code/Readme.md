# KL-aggregation
This repository contains code written during my Ph.D thesis for "Optimal Kullback-Leibler Aggregation in Mixture Density Estimation by Maximum Likelihood" paper.
https://www.ems-ph.org/journals/show_pdf.php?issn=2520-2316&vol=1&iss=1&rank=1

Note: this code is provided 'as-is' and need refactoring to be used. It relies heavily on Numba, comment @jit decorator for testing. 

Several portions of this code come from different sources, some of which I did not referenced properly, nevertheless, I thank the authors.

the models present in this repository are:

- KL estimator for Mixture density estimation
- Experiments used on my thesis (https://pastel.archives-ouvertes.fr/tel-01677233)

I also implemented
- Adaptitve Dantzig estimator from from K. Bertin, E. Le Pennec and V. Rivoirard
- SPADES estimator, by Florentina Bunea, Alexandre B. Tsybakov, Marten H. Wegkamp and Adrian Barbu
