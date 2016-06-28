cimport cython
from libc.math cimport pow, sqrt
from libc.stdlib cimport strtoul, strtod, malloc, realloc, free
from libc.stdio cimport fopen, fclose, FILE, fprintf

ctypedef unsigned long ULong


cdef struct FeatureVector:
    int size
    ULong * index
    double * value


cdef struct DataSet:
    ULong size
    FeatureVector * features


cdef struct Similarity:
    ULong size
    double * value


# This function will run within Python GIL to utilize yield.
# The compromise in speed is to accomodate for reading one line at a time from
# the file instead of trying to load the entire file in the memory
def fileIter(str filename):
    cdef str line
    with open(filename) as ifi:
        for line in ifi:
            yield line


cdef FeatureVector * parseVector(bytes vector_string):
    cdef bytes index, feature_val
    cdef int i
    cdef char * tmp
    feature_strings = vector_string.split()
    cdef int num_features = len(feature_strings)
    cdef FeatureVector * features = <FeatureVector *>malloc(
        sizeof(FeatureVector)
    )
    if not features:
        raise MemoryError()
    features.index = <ULong *>malloc(sizeof(ULong) *
                                                num_features)
    features.value = <double *>malloc(sizeof(double) * num_features)
    if not features.index or not features.value:
        raise MemoryError()
    features.size = num_features
    for i in range(num_features):
        index, feature_val = feature_strings[i].split(':')
        features.index[i] = strtoul(index, &tmp, 10)
        features.value[i] = strtod(feature_val, &tmp)
    return features


cdef DataSet * getAllVectors(str filename):
    """
    Since the total number of vectors is not known, the file will be read in
    batches and memory for the struct will be reallocated successively for each
    batch.
    """
    cdef DataSet * all_vectors = <DataSet *>malloc(sizeof(DataSet))
    if not all_vectors:
        raise MemoryError()
    cdef bytes vector_string
    cdef ULong batch_size = 1000
    cdef ULong batch_counter = 0
    # Initial allocation for dataset
    # The pointer needs to be null for realloc to not bork
    all_vectors.features = NULL
    all_vectors.size = 0
    file_reader = fileIter(filename)
    cdef bint end_of_file = 0
    while not end_of_file:
        batch_counter += 1
        all_vectors.features = <FeatureVector *>realloc(
            all_vectors.features,
            sizeof(FeatureVector) * batch_size * batch_counter
        )
        if not all_vectors.features:
            print "Ran out of memory while reading all vectors"
            raise MemoryError()
        for i in range(batch_size):
            try:
                vector_string = file_reader.next()
            except StopIteration:
                batch_size = i
                end_of_file = 1
                break
            vector_index = (batch_counter - 1) * batch_size + i
            all_vectors.features[vector_index] = parseVector(vector_string)[0]
        all_vectors.size += batch_size
    return all_vectors


cdef double getVectorLength(FeatureVector * vector):
    cdef double v_length = 0.0
    cdef int idx
    for idx in range(vector.size):
        v_length += pow(vector.value[idx], 2)
    return sqrt(v_length)


cdef double getSimilarity(FeatureVector * vector1, FeatureVector * vector2):
    if vector1.size > vector2.size:
        vector1, vector2 = vector2, vector1

    cdef double dot_prod = 0.0
    cdef int v1_idx, v2_idx
    for v1_idx in range(vector1.size):
        for v2_idx in range(vector2.size):
            if vector2.index[v2_idx] == vector1.index[v1_idx]:
                dot_prod += vector1.value[v1_idx] * vector2.value[v2_idx]
                break
    return dot_prod / (getVectorLength(vector1) * getVectorLength(vector2))


# @cython.cdivision(True)
# @cython.wraparound(False)
# @cython.boundscheck(False)
cdef Similarity * buildSimilarityMatrix(DataSet * all_vectors):
    cdef ULong i, j
    cdef bytes matrix_key
    cdef ULong num_entries = <ULong>((pow(all_vectors.size, 2) -
                                      all_vectors.size) / 2)
    cdef Similarity * sim_matrix = <Similarity *>malloc(sizeof(Similarity))
    if not sim_matrix:
        raise MemoryError()
    sim_matrix.value = <double *>malloc(sizeof(double) * num_entries)
    sim_matrix.size = num_entries
    if not sim_matrix.value:
        raise MemoryError()
    cdef ULong counter = 0
    for i in range(all_vectors.size - 1):
        for j in range(i + 1, all_vectors.size):
            matrix_key = str(i) + '-' + str(j)
            sim_matrix.value[counter] = getSimilarity(&all_vectors.features[i],
                                                      &all_vectors.features[j])
            counter += 1
    return sim_matrix


cdef printSimilarityMatrix(Similarity * sim_matrix, ULong size,  str outfile):
    cdef int idx
    cdef FILE * f = fopen(outfile, "w")
    for idx in range(sim_matrix.size):
        fprintf(f, "%f\n", sim_matrix.value[idx])
    fclose(f)


def main():
    cdef str infile = './3k.vec'
    cdef str outfile = './3k.sim'
    cdef DataSet * all_vectors = getAllVectors(infile)
    # print 'Collected all {} vectors'.format(all_vectors.size)
    sim_matrix = buildSimilarityMatrix(all_vectors)
    # print 'Built Similarity Matrix'
    # printSimilarityMatrix(sim_matrix, all_vectors.size, outfile)
    # print 'Similarity Matrix saved'
    free(all_vectors.features.index)
    free(all_vectors.features.value)
    free(all_vectors.features)
    free(all_vectors)
    free(sim_matrix.value)
    free(sim_matrix)
