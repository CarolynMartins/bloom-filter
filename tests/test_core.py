import math
import unittest

from bloom_filter import BloomFilter


class TestConstruction(unittest.TestCase):
    def test_default_error_rate(self):
        bf = BloomFilter(1000)
        self.assertEqual(bf.capacity, 1000)
        self.assertEqual(bf.error_rate, 0.01)
        self.assertEqual(bf.count, 0)

    def test_rejects_zero_capacity(self):
        with self.assertRaises(ValueError):
            BloomFilter(0)

    def test_rejects_negative_capacity(self):
        with self.assertRaises(ValueError):
            BloomFilter(-5)

    def test_rejects_non_integer_capacity(self):
        with self.assertRaises(ValueError):
            BloomFilter(100.0)

    def test_rejects_bool_capacity(self):
        # bool is a subclass of int; we treat it as invalid because a capacity
        # of True/False is almost certainly a caller bug.
        with self.assertRaises(ValueError):
            BloomFilter(True)

    def test_rejects_error_rate_zero(self):
        with self.assertRaises(ValueError):
            BloomFilter(100, 0.0)

    def test_rejects_error_rate_one(self):
        with self.assertRaises(ValueError):
            BloomFilter(100, 1.0)

    def test_rejects_error_rate_out_of_range(self):
        with self.assertRaises(ValueError):
            BloomFilter(100, 1.5)
        with self.assertRaises(ValueError):
            BloomFilter(100, -0.1)


class TestBasicOperations(unittest.TestCase):
    def test_added_item_is_present(self):
        bf = BloomFilter(100)
        bf.add(b"hello")
        self.assertIn(b"hello", bf)

    def test_absent_item_may_be_absent(self):
        bf = BloomFilter(100)
        # An empty filter must report every item as absent.
        self.assertNotIn(b"anything", bf)

    def test_count_tracks_adds(self):
        bf = BloomFilter(100)
        self.assertEqual(len(bf), 0)
        bf.add(b"a")
        bf.add(b"b")
        self.assertEqual(len(bf), 2)

    def test_add_returns_item(self):
        bf = BloomFilter(10)
        result = bf.add(b"x")
        self.assertEqual(result, b"x")

    def test_rejects_non_bytes_item(self):
        bf = BloomFilter(10)
        with self.assertRaises(TypeError):
            bf.add("not bytes")
        with self.assertRaises(TypeError):
            "not bytes" in bf


class TestNoFalseNegatives(unittest.TestCase):
    """The core guarantee: every added item must be reported as present."""

    def test_all_added_items_present(self):
        bf = BloomFilter(500, 0.01)
        items = [str(i).encode() for i in range(500)]
        for item in items:
            bf.add(item)
        for item in items:
            self.assertIn(item, bf)

    def test_repeated_adds_keep_item_present(self):
        bf = BloomFilter(10)
        bf.add(b"dup")
        bf.add(b"dup")
        bf.add(b"dup")
        self.assertIn(b"dup", bf)
        self.assertEqual(len(bf), 3)


class TestFalsePositiveRate(unittest.TestCase):
    """Verify the observed false positive rate is within the expected bound.

    We test at the configured capacity, where the rate should be at most
    error_rate. We allow a generous margin (3x) because Bloom filter analysis
    is asymptotic and the optimal-k formula is a ceiling, not an exact bound.
    """

    def test_false_positive_rate_under_bound(self):
        capacity = 2000
        error_rate = 0.01
        bf = BloomFilter(capacity, error_rate)
        for i in range(capacity):
            bf.add(f"item-{i}".encode())

        # Probe with items that were never added.
        probes = 5000
        false_positives = 0
        for i in range(probes):
            probe = f"absent-{i}".encode()
            if probe in bf:
                false_positives += 1

        observed = false_positives / probes
        # 3x margin: the analysis is approximate and we want this test to be
        # deterministic across runs and platforms.
        self.assertLess(observed, error_rate * 3)


class TestInternalLayout(unittest.TestCase):
    def test_num_bits_and_hashes_are_positive(self):
        bf = BloomFilter(100, 0.01)
        self.assertGreater(bf.num_bits, 0)
        self.assertGreaterEqual(bf.num_hashes, 1)

    def test_bytearray_size_matches_num_bits(self):
        bf = BloomFilter(100, 0.01)
        expected_bytes = (bf.num_bits + 7) // 8
        self.assertEqual(len(bf._bits), expected_bytes)

    def test_repr_contains_key_fields(self):
        bf = BloomFilter(100, 0.05)
        text = repr(bf)
        self.assertIn("BloomFilter(", text)
        self.assertIn("capacity=100", text)
        self.assertIn("count=0", text)


class TestHighHashCount(unittest.TestCase):
    """Exercise the path where k exceeds 8, forcing a digest re-hash."""

    def test_large_k_uses_rehashing(self):
        # A very low error rate on a modest capacity pushes k well above 8,
        # which exercises the re-hashing branch in _positions.
        bf = BloomFilter(50, 0.0001)
        self.assertGreater(bf.num_hashes, 8)
        bf.add(b"trigger")
        self.assertIn(b"trigger", bf)


class TestTinyCapacity(unittest.TestCase):
    def test_capacity_one(self):
        bf = BloomFilter(1, 0.5)
        bf.add(b"only")
        self.assertIn(b"only", bf)


if __name__ == "__main__":
    unittest.main()
