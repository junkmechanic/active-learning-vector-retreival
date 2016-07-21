import os
import sys
import json

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdlib cimport strtod

from similarity import fileIter

from similarity cimport DataSet, Similarity
from similarity cimport getAllVectors, buildSimilarityMatrix
from clustering cimport buildClusters

ctypedef unsigned long ULong


cdef double * getConfidenceScores(filename, data_size):
    cdef:
        bytes line
        ULong i
        char * tmp
        double * confidences = <double *> PyMem_Malloc(sizeof(double) *
                                                       data_size)
    if not confidences:
        raise MemoryError()
    i = 0
    for line in fileIter(filename):
        confidences[i] = strtod(line.strip(), &tmp)
        i += 1
    if i != data_size:
        print "Number of confidence scores ({}) != Number of vectors ({})".\
            format(i, data_size)
        PyMem_Free(confidences)
        return NULL
    return confidences


cdef getMedoids(
    ULong data_size,
    Similarity * sim_matrix,
    recompute_clusters,
    int num_clusters,
    int cluster_max_iterations,
    int cluster_patience,
    unsigned int seed,
):
    cdef:
        ULong i
        str medoid_file = './data/medoids.json'
        ULong * medoids = <ULong *> PyMem_Malloc(sizeof(ULong) * num_clusters)
        ULong * assigned = <ULong *> PyMem_Malloc(sizeof(ULong) * data_size)
    if not medoids or not assigned:
        raise MemoryError()

    # The file is a dictionary with two keys : `medoids` and `densities`
    if os.path.exists(medoid_file):
        with open(medoid_file) as ifi:
            stats = json.load(ifi)
            medoid_list, densities = stats['medoids'], stats['densities']
        if len(medoid_list) != num_clusters:
            recompute_clusters = True
    else:
        recompute_clusters = True

    if recompute_clusters:
        medoid_list = []
        densities = [0] * num_clusters
        print "Running clustering to find {} medoids".format(num_clusters)
        buildClusters(
            sim_matrix,
            medoids,
            assigned,
            data_size,
            num_clusters,
            cluster_max_iterations,
            cluster_patience,
            seed
        )
        for i in range(num_clusters):
            medoid_list.append(medoids[i])
        for i in range(data_size):
            densities[assigned[i]] += 1
        stats = {'medoids': medoid_list, 'densities': densities}
        print "Saving in {}".format(medoid_file)
        with open(medoid_file, 'w') as ofi:
            json.dump(stats, ofi)
        PyMem_Free(medoids)
        PyMem_Free(assigned)
    else:
        print "Using pre-saved medoids file {}".format(medoid_file)
    return medoid_list, densities


def getNewVectors(
    str vector_file,
    str confidence_file,
    int number_candidates=100,
    double weight_confidence=0.5,
    double weight_density=0.5,
    recompute_clusters=False,
    int num_clusters=300,
    int cluster_max_iterations=500,
    int cluster_patience=20,
    unsigned int seed = 200816,
):
    cdef int i
    cdef DataSet * all_vectors = getAllVectors(vector_file)
    cdef Similarity * sim_matrix = buildSimilarityMatrix(all_vectors)
    print "Built Similarity Matrix"
    medoids, densities = getMedoids(all_vectors.size, sim_matrix,
                                    recompute_clusters, num_clusters,
                                    cluster_max_iterations, cluster_patience,
                                    seed)
    # `clusters` is a contains medoid indices as keys and the corresponding
    # densities as values
    clusters = {}
    for medoid, density in zip(medoids, densities):
        if medoid in clusters:
            clusters[medoid]['density'] += density
        else:
            clusters[medoid] = {'density': density}
    print "{} candidates collected.".format(len(clusters))

    cdef double * confidences = getConfidenceScores(confidence_file,
                                                    all_vectors.size)
    if not confidences:
        PyMem_Free(all_vectors)
        PyMem_Free(sim_matrix)
        sys.exit(1)

    PyMem_Free(all_vectors)
    PyMem_Free(sim_matrix)
    PyMem_Free(confidences)

    low, high = 0, 0
    for medoid in clusters:
        clusters[medoid]['confidence'] = confidences[medoid]
        if clusters[medoid]['density'] < low:
            low = clusters[medoid]['density']
        if clusters[medoid]['density'] > high:
            high = clusters[medoid]['density']
    for medoid in clusters:
        clusters[medoid]['density'] = ((clusters[medoid]['density'] - low) /
                                       float(high - low))
        clusters[medoid]['measure'] = (
            weight_confidence * (1 - clusters[medoid]['confidence']) +
            weight_density * clusters[medoid]['density']
        )

    sorted_meds = sorted(clusters, key=lambda x: clusters[x]['measure'])
    return sorted_meds[:number_candidates]


# Assumptions
# the confidence file should contain as many number of results as there are
# input vectors
# the confidence file should only contain one double value per line each
# idicating the confidence of the vector at that index in the input file
