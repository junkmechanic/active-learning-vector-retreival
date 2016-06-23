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
        v_length += vector[idx] ** 2
    return v_length ** 0.5


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


def printSimilarityMatrix(sim_matrix, size, filename):
    # The dictionary (hashmap) representing the similarity matrix does not
    # contain values for the diagnal because the diagnal vector is an identity
    # vector.
    with open(filename, 'w') as ofi:
        for i in range(size):
            sim_vector = ''
            for j in range(size):
                if i == j:
                    sim_vector += str(float(1))
                else:
                    key1, key2 = '{}-{}'.format(i, j), '{}-{}'.format(j, i)
                    key = key1 if key1 in sim_matrix else key2
                    sim_vector += str(sim_matrix[key])
                if j != (size - 1):
                    sim_vector += '\t'
            ofi.write(sim_vector + '\n')


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
