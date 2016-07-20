from libc.stdlib cimport rand, srand
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from cython.parallel import parallel, prange

from similarity cimport DataSet, Similarity, getSimilarity
from similarity cimport getAllVectors, buildSimilarityMatrix

ctypedef unsigned long ULong


cdef void doAssignments(
    ULong * medoids,
    ULong * assigned,
    Similarity * sim_matrix,
    ULong data_size,
    int num_clusters
):
    """
    Assignes each sample to its closest (in terms of similarity) medoid
    """
    cdef:
        int j
        ULong i, med_idx
        double sim, max_sim
    with nogil, parallel():
        for i in prange(data_size):
            # each sample i will be compared with each medoid and the one with
            # most similarity will be assigned
            max_sim = 0.0
            med_idx = 0
            for j in range(num_clusters):
                sim = getSimilarity(sim_matrix, i, medoids[j], data_size)
                if sim > max_sim:
                    max_sim = sim
                    med_idx = j
            assigned[i] = med_idx


cdef void fillClusterMap(
    ULong ** cluster_map,
    ULong * cluster_idx,
    ULong * medoids,
    ULong * assigned,
    ULong data_size,
    int num_clusters
):
    """
    Populates a cluster map from given assigned array based on the medoids along
    with cluster_idx for each cluster.
    The two need to be passed.
    """
    cdef ULong i, j, c_idx, c_ptr
    for i in range(num_clusters):
        cluster_idx[i] = 1

    # First we need to know how many samples belong to each cluster. Only then
    # can we allocate memory for each cluster.
    for i in range(data_size):
        c_idx = assigned[i]
        cluster_idx[c_idx] += 1

    # Now we can assign memory for each cluster
    for i in range(num_clusters):
        cluster_map[i] = <ULong *> PyMem_Malloc(sizeof(ULong) *
                                                cluster_idx[i])
        if not cluster_map[i]:
            raise MemoryError()

    # Reinitialize cluster_idx to help keep track of cluster indices during map
    # population
    for i in range(num_clusters):
        cluster_idx[i] = 0

    # Now to the actual map population

    # NOTE: This is faster than the one used below but this is not
    # parallelizable
    # for i in range(data_size):
    #     c_idx = assigned[i]
    #     c_ptr = cluster_idx[c_idx]
    #     cluster_map[c_idx][c_ptr] = i
    #     cluster_idx[c_idx] += 1

    with nogil, parallel():
        for i in prange(num_clusters):
            for j in range(data_size):
                if assigned[j] == i:
                    c_ptr = cluster_idx[i]
                    cluster_map[i][c_ptr] = j
                    cluster_idx[i] = cluster_idx[i] + 1


cdef double compute_clusters(
    ULong * medoids,
    ULong * assigned,
    Similarity * sim_matrix,
    ULong data_size,
    int num_clusters,
):
    """
    Computes new cluster medoids based on the cosine similarity.
    The objective function to maximize is the sum of cosine similarities between
    the current cluster medoid and each of the samples belonging to the
    corresponding cluster.
    This is done by greedily searching through the entire cluster to find the
    point that would maximize this sum.

    First the 2-d array `cluster_map` is populated with each cluster
    corresponding to the `medoids`.
    Then the maximize sum is calculated.
    """
    cdef:
        int i
        ULong j, k, med_idx
        double max_sum, running_sum, cluster_sum
        ULong ** cluster_map = <ULong **> PyMem_Malloc(sizeof(ULong *) *
                                                       num_clusters)
        ULong * cluster_idx = <ULong *> PyMem_Malloc(sizeof(ULong) *
                                                     num_clusters)
    if not cluster_map or not cluster_idx:
        raise MemoryError()

    fillClusterMap(cluster_map, cluster_idx, medoids, assigned, data_size,
                   num_clusters)

    running_sum = 0.0
    with nogil, parallel():
        for i in prange(num_clusters):
            max_sum = 0.0
            med_idx = 0
            for j in range(cluster_idx[i]):
                cluster_sum = 0.0
                for k in range(cluster_idx[i]):
                    cluster_sum = cluster_sum + getSimilarity(sim_matrix,
                                                              cluster_map[i][j],
                                                              cluster_map[i][k],
                                                              data_size)
                if cluster_sum > max_sum:
                    max_sum = cluster_sum
                    med_idx = cluster_map[i][j]
            medoids[i] = med_idx
            running_sum += max_sum

    doAssignments(medoids, assigned, sim_matrix, data_size, num_clusters)

    # Clean up
    PyMem_Free(cluster_idx)
    for i in range(num_clusters):
        PyMem_Free(cluster_map[i])
    PyMem_Free(cluster_map)

    return running_sum


cdef void buildClusters(
    Similarity * sim_matrix,
    ULong * medoids,
    ULong * assigned,
    ULong data_size,
    int num_clusters,
    int max_iterations,
    int patience,
):
    """
    medoids : array containing the indices of the medoids
              size = num_clusters
              range of values = 0 to `data_size - 1`
    assigned : array containing the indices of the medoid that the corresponding
               sample is closest to. The index indicates the cluster index of
               the array `medoids`
               size = data_size
               range of values = 0 to `num_clusters - 1`
    """
    cdef:
        int i, j, hist, itr = 0
        unsigned int seed = 200816
        ULong rand_idx
        bint unique, converged
        double * entropy = <double *> PyMem_Malloc(sizeof(double) *
                                                   max_iterations)
    if not entropy:
        raise MemoryError()

    # Initialize medoids with random samples
    srand(seed)
    for i in range(num_clusters):
        rand_idx = rand() % data_size
        # ensure this has not already been chosen as a medoid
        while True:
            unique = 1
            for j in range(i):
                if medoids[j] == rand_idx:
                    rand_idx = rand() % data_size
                    unique = 0
                    break
            if unique:
                break
        medoids[i] = rand_idx

    # Do clustering till convergence or for max_iterations
    doAssignments(medoids, assigned, sim_matrix, data_size, num_clusters)
    converged = 0
    while not converged and itr < max_iterations:
        entropy[itr] = compute_clusters(medoids, assigned, sim_matrix,
                                        data_size, num_clusters)
        print("Iteration : %d , Entropy : %f"%(itr, entropy[itr]))
        # Since it is a variant of k-medoid clustering, chances are that the
        # medoids oscilate within a cluster on convergence. So we can't just
        # check for repretition of the same entropy. Rather we need to check for
        # repeating patterns. That could take polynomial time and it should be
        # enough to check if the last value repeats itself `patience` number of
        # times.
        hist = 0
        for i in range(itr - 1, 0, -1):
            if entropy[i] == entropy[itr]:
                hist += 1
                if hist > patience:
                    converged = 1
        itr += 1

    if converged:
        print "Clustering converged after %d iterations"%(itr)
    else:
        print "Maximum number of iterations reached : %d"%(max_iterations)

    # Clean up
    PyMem_Free(entropy)


def test_clustering(int num_clusters=200, int max_iterations=500):
    cdef str infile = './data/80k.vec'
    cdef DataSet * all_vectors = getAllVectors(infile)
    cdef Similarity * sim_matrix = buildSimilarityMatrix(all_vectors)
    print "Built Similarity Matrix"

    cdef patience = 20
    cdef ULong * medoids = <ULong *> PyMem_Malloc(sizeof(ULong) * num_clusters)
    cdef ULong * assigned = <ULong *> PyMem_Malloc(sizeof(ULong) *
                                                    all_vectors.size)
    if not medoids or not assigned:
        raise MemoryError()
    buildClusters(
        sim_matrix,
        medoids,
        assigned,
        all_vectors.size,
        num_clusters,
        max_iterations,
        patience
    )
    PyMem_Free(medoids)
    PyMem_Free(assigned)
