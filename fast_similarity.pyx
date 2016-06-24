from libc.math cimport pow, sqrt


def fileIter(str filename):
    cdef str line
    with open(filename) as ifi:
        for line in ifi:
            yield line


def parseVector(str vector_string):
    vector = {}
    cdef str feature_string, index, feature_val
    for feature_string in vector_string.split():
        index, feature_val = feature_string.split(':')
        vector[int(index)] = float(feature_val)
    return vector


def getAllVectors(str filename):
    all_vectors = {}
    cdef int idx
    cdef str vector_string
    for idx, vector_string in enumerate(fileIter(filename)):
        vector = parseVector(vector_string)
        all_vectors[idx] = vector
    return all_vectors


cdef double getVectorLength(vector):
    cdef double v_length = 0.0
    cdef int idx
    for idx in vector:
        v_length += pow(vector[idx], 2)
    return sqrt(v_length)


cdef double getSimilarity(vector1, vector2):
    if len(vector1) > len(vector2):
        vector1, vector2 = vector2, vector1

    cdef double dot_prod = 0.0
    cdef int idx
    for idx in vector1:
        if idx in vector2:
            dot_prod += vector1[idx] * vector2[idx]
    return dot_prod / (getVectorLength(vector1) * getVectorLength(vector2))


def buildSimilarityMatrix(all_vectors):
    sim_matrix = {}
    cdef int i
    cdef int num_vectors = len(all_vectors)
    cdef str matrix_key
    all_idxs = all_vectors.keys()
    for i in range(num_vectors - 1):
        for j in range(i + 1, num_vectors):
            matrix_key = '{}-{}'.format(all_idxs[i], all_idxs[j])
            sim_matrix[matrix_key] = getSimilarity(all_vectors[i],
                                                   all_vectors[j])
    return sim_matrix


def main():
    cdef str infile = './3k.vec'
    cdef str outfile = './3k.sim'
    all_vectors = getAllVectors(infile)
    print 'Collected all {} vectors'.format(len(all_vectors))
    sim_matrix = buildSimilarityMatrix(all_vectors)
    print 'Built Similarity Matrix'
    # printSimilarityMatrix(sim_matrix, len(all_vectors), outfile)
    # print 'Similarity Matrix saved'


# if __name__ == '__main__':
#     main()
