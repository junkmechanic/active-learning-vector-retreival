from libc.stdlib cimport rand, srand
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

from similarity cimport DataSet, Similarity, FeatureVector, getSimilarity
from similarity cimport getAllVectors, buildSimilarityMatrix

ctypedef unsigned long ULong
ctypedef long long LLong


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
        ULong i
        double max_sim
        med_idx
    for i in range(data_size):
        # each sample i will be compared with each medoid and the one with most
        # similarity will be assigned
        max_sim = 0.0
        for j in range(num_clusters):
            sim = getSimilarity(sim_matrix, i, medoids[j], data_size)
            if sim > max_sim:
                max_sim = sim
                med_idx = j
        assigned[i] = j


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
    cdef ULong i
    for i in range(num_clusters):
        cluster_idx[i] = 0

    for i in range(data_size):
        c_idx = assigned[i]
        cluster_map[c_idx][cluster_idx[c_idx]] = i
        cluster_idx[c_idx] += 1


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
        ULong j, k
        double max_sum, running_sum, cluster_sum
        ULong med_idx
        ULong ** cluster_map = <ULong **> PyMem_Malloc(sizeof(ULong *) *
                                                       num_clusters)
        ULong * cluster_idx = <ULong *> PyMem_Malloc(sizeof(ULong) *
                                                     num_clusters)

    fillClusterMap(cluster_map, cluster_idx, medoids, assigned, data_size,
                   num_clusters)

    running_sum = 0.0
    for i in range(num_clusters):
        max_sum = 0.0
        for j in range(cluster_idx[i]):
            cluster_sum = 0.0
            for k in range(cluster_idx[i]):
                cluster_sum += getSimilarity(sim_matrix, cluster_map[i][j],
                                             cluster_map[i][k], data_size)
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
        int i, j, itr = 0, hist = 0
        unsigned int seed = 2016
        ULong rand_idx
        float entropy
        bint unique, converged
        ULong * prev_medoids = <ULong *> PyMem_Malloc(sizeof(ULong) *
                                                      num_clusters)

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
        prev_medoids[i] = rand_idx

    # Do clustering till convergence or for max_iterations
    doAssignments(medoids, assigned, sim_matrix, data_size, num_clusters)
    converged = 0
    while not converged or itr < max_iterations:
        entropy = compute_clusters(medoids, assigned, sim_matrix, data_size,
                                   num_clusters)
        print("Iteration : %d , Entropy : %f"%(itr, entropy))
        changed = 0
        for i in range(num_clusters):
            if medoids[i] != prev_medoids[i]:
                changed = 1
                break
        if not changed:
            hist += 1
        else:
            hist = 0
            for i in range(num_clusters):
                prev_medoids[i] = medoids[i]
        if hist == patience:
            converged = 1

    # Clean up
    PyMem_Free(prev_medoids)


def test_clustering():
    cdef str infile = './data/3k.vec'
    cdef DataSet * all_vectors = getAllVectors(infile)
    cdef Similarity * sim_matrix = buildSimilarityMatrix(all_vectors)
    print 'Built Similarity Matrix'

    cdef int num_clusters = 200, max_iterations = 10000, patience = 20
    cdef ULong * medoids = <ULong *> PyMem_Malloc(sizeof(ULong) * num_clusters)
    cdef ULong * assigned = <ULong *> PyMem_Malloc(sizeof(ULong) *
                                                    all_vectors.size)
    buildClusters(
        sim_matrix,
        medoids,
        assigned,
        all_vectors.size,
        num_clusters,
        max_iterations,
        patience
    )
    print "Built Clusters"
