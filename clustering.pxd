ctypedef unsigned long ULong

from similarity cimport Similarity


cdef void buildClusters(
    Similarity * sim_matrix,
    ULong * medoids,
    ULong * assigned,
    ULong data_size,
    int num_clusters,
    int max_iterations,
    int patience,
    unsigned int seed,
)
