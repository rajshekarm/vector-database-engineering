
from src.dataset import VectorDataset


def test_vector_shape():

    dataset = VectorDataset(10000, 384)

    vectors, ids = dataset.generate()

    assert vectors.shape == (10000, 384)
    assert len(ids) == 10000


def test_vector_datatype():

    dataset = VectorDataset()

    vectors, _ = dataset.generate()

    assert str(vectors.dtype) == "float32"


def test_query_dimensions():

    dataset = VectorDataset()

    query = dataset.generate_query()

    assert query.shape == (384,)


def test_reproducibility():

    dataset1 = VectorDataset(seed=42)
    dataset2 = VectorDataset(seed=42)

    vectors1, _ = dataset1.generate()
    vectors2, _ = dataset2.generate()

    assert (vectors1 == vectors2).all()
