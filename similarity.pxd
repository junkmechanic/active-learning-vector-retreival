ctypedef unsigned long ULong
ctypedef long long LLong


# Struct Declarations

cdef struct FeatureVector:
    int size
    ULong * index
    double * value

cdef struct DataSet:
    ULong size
    FeatureVector * features

cdef struct Similarity:
    LLong size
    double * value


# Function Declarations

cdef DataSet * getAllVectors(str filename)
cdef Similarity * buildSimilarityMatrix(DataSet * all_vectors)
cdef double getSimilarity(
    Similarity * sim_matrix,
    ULong idx1,
    ULong idx2,
    ULong data_size
)
