
import numpy as np

def generate_sample_f_star(N, t="uniform", densities=None, selected_densities=None, s=5, cvx_rand=False, dist_rect=None,
                           N_pdf=1000):
    # Generate a sample of n points in [0,1]
    # t can be uniform, convex, rect, in the case of convex, a convex combination of elements of gaussian
    # densities will be taken with a sparsity s
    # we return also return the distribution on [0,1], 1000 points
    if t == "uniform":
        X_ = np.random.uniform(low=0, high=1, size=N)
        return X_, 1. / N_pdf * np.ones(N_pdf)
    if t == "convex":
        if densities == None:
            raise ValueError("Densities were not given")

        def generate_points(N, s, densities, selected_densities, sample_repartition_among_clusters, weights):
            print "selected densities: ", sorted(zip(selected_densities, weights))
            X = np.array([])
            t = 0
            # We generate the sample according to the selected densities and the weights
            for i in range(s):
                X = np.hstack((X, densities[selected_densities[i]].rvs(sample_repartition_among_clusters[i])))
            np.random.shuffle(X)
            return X, np.apply_along_axis(
                lambda x: weights.dot(np.array([densities[i].pdf(x) for i in selected_densities])), 0,
                np.linspace(0, 1, N_pdf)), weights, selected_densities

        # We select randomly s elements in densities, then we have two possibilities depending on cvx_rand:
        # false: equal weight for each densities, w=1/s
        # true: random weights
        if selected_densities == None:
            selected_densities = np.random.choice(len(densities), s, replace=False)
        s = len(selected_densities)
        if cvx_rand:
            # We generate the weights
            weights = np.random.randint(N * 100, size=(1, s))[0]
            weights = 1. * weights / weights.sum()
            sample_repartition_among_clusters = np.random.multinomial(N, weights, size=1)[0]
            return generate_points(N, s, densities, selected_densities, sample_repartition_among_clusters, weights), weights
        else:
            a = round(N / s) * np.ones(s)
            # We adjust the size to the last element
            a[-1] = a[-1] - (a.sum() - N)
            sample_repartition_among_clusters = a.astype(int)
            return generate_points(N, s, densities, selected_densities, sample_repartition_among_clusters, a / a.sum()), a / a.sum()
    if t == "rect":
        def prob_estim(x, dist_rect):
            for intval in sorted(dist_rect.keys()):
                if x <= intval[1]:
                    return dist_rect[intval]

        X_ = np.linspace(0, 1, 10 * N)
        proba = np.array([prob_estim(x, dist_rect) for x in X_])
        proba = proba / proba.sum()
        return np.random.choice(X_, size=N, p=proba), np.array(
            [prob_estim(x, dist_rect) for x in np.linspace(0, 1, N_pdf)])






