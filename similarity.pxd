# Struct Declarations

cdef struct FeatureVector
cdef struct DataSet
cdef struct Similarity

# Function Declarations

cdef DataSet * getAllVectors(str filename)
cdef Similarity * buildSimilarityMatrix(DataSet * all_vectors)
