
import numpy as np


class VectorDataset:

    def __init__(
        self,
        num_vectors=10000,
        dimensions=384,
        seed=42
    ):
        self.num_vectors = num_vectors
        self.dimensions = dimensions
        self.seed = seed

        self.vectors = None
        self.ids = None

    def generate(self):

        rng = np.random.default_rng(self.seed)

        self.vectors = rng.random(
            (self.num_vectors, self.dimensions),
            dtype=np.float32
        )

        self.ids = np.arange(self.num_vectors)

        return self.vectors, self.ids

    def generate_query(self):

        rng = np.random.default_rng(self.seed + 1)

        query = rng.random(
            self.dimensions,
            dtype=np.float32
        )

        return query


if __name__ == "__main__":

    dataset = VectorDataset()

    vectors, ids = dataset.generate()

    query = dataset.generate_query()

    print("Dataset generated successfully!")
    print("Vector shape:", vectors.shape)
    print("ID count:", len(ids))
    print("Query shape:", query.shape)
    print("Data type:", vectors.dtype)
