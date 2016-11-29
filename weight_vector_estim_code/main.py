from densities import generate_sample
from algorithm import WeightEstimator

X, densities_dict, selected_densities, weights = generate_sample(1000, dim=5, n_densities=10, K=100, var=10**(-3))
select_weighted_densities = zip(selected_densities, weights)
select_weighted_densities.sort(key=lambda x: x[0])

cl = WeightEstimator(densities_dict=densities_dict)
cl.fit(X)
estim_weighted_densities=cl.select_densities()

print "vraies densites: ", select_weighted_densities
print "densites estimees", estim_weighted_densities
