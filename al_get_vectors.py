from choose_vectors import getNewVectors

suffix = '3k'
# suffix = '80k'
vector_file = './data/' + suffix + '.vec'
confidence_file = './data/test_confidences_' + suffix + '.out'
new_vectors = getNewVectors(vector_file,
                            confidence_file,
                            recompute_clusters=False,
                            num_clusters=300,)
print new_vectors
