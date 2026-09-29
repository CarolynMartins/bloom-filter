"""Bloom filter implementation using only the standard library.

Design decisions:

- We use MD5 as the hash function. It is not cryptographically strong, but
  that is irrelevant here; we need a fast, well-distributed hash whose output
  we can slice into k independent bit positions. MD5 is in the standard library
  on every supported Python version, unlike blake2b which is also fine but not
  worth the extra import surface for this use case.

- We derive k hashes from a single MD5 digest by taking 16-bit slices and
  reducing each modulo the bit-array length. Double hashing (k = h1 + i*h2) is
  the classic space-saver, but it introduces correlation when the filter is
  small relative to k. Slicing the digest avoids that correlation at the cost
  of one MD5 call per add/lookup, which is negligible for the sizes this
  library targets.

- The bit array is a bytearray. A Python int would let us set bits with |=, but
  a bytearray keeps memory predictable and lets us serialise with bytes()
  without serialisation surprises across Python versions.
"""

import hashlib
import math


class BloomFilter:
    """A space-efficient set membership test with a tunable false positive rate.

    A Bloom filter can answer "is this item definitely not in the set?" with
    certainty, and "is this item in the set?" with a probability of being wrong
    that is bounded by the configured false positive rate.

    Items must be bytes. Callers are responsible for encoding strings; forcing
    the caller to be explicit about encoding avoids the classic bug where the
    same string added as UTF-8 and looked up as Latin-1 produces inconsistent
    results.
    """

    def __init__(self, capacity, error_rate=0.01):
        """Create a new, empty Bloom filter.

        Args:
            capacity: Maximum number of items the filter should hold while
                still meeting error_rate. Adding more items than this raises
                the actual false positive rate above error_rate.
            error_rate: Desired upper bound on the false positive probability
                when the filter holds exactly ``capacity`` items. Must be in
                the open interval (0, 1).

        Raises:
            ValueError: If capacity is not a positive integer or error_rate is
                outside (0, 1).
        """
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            raise ValueError("capacity must be a positive integer")
        if capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if not (0.0 < error_rate < 1.0):
            raise ValueError("error_rate must be in the open interval (0, 1)")

        self.capacity = capacity
        self.error_rate = error_rate

        # Optimal bit count: m = -(n * ln(p)) / (ln(2)^2)
        m = math.ceil(-((capacity * math.log(error_rate)) / (math.log(2) ** 2)))
        # Guard against m == 0, which happens for tiny capacities and loose
        # error rates. A zero-length filter can never record anything.
        if m < 1:
            m = 1
        self.num_bits = m

        # Optimal hash count: k = (m / n) * ln(2)
        k = round((m / capacity) * math.log(2))
        if k < 1:
            k = 1
        self.num_hashes = k

        # bytearray is 8 bits per element; ceil division gives us enough bytes.
        self._bits = bytearray((m + 7) // 8)
        self.count = 0

    def _positions(self, item):
        """Yield the k bit positions for ``item``.

        We hash the item once with MD5 to get 16 bytes, then slice those bytes
        into 16-bit integers. Each slice is reduced modulo num_bits to land in
        the valid range. If k exceeds 8 (the number of 16-bit slices in a
        128-bit digest) we re-hash the digest and keep slicing; this keeps the
        positions independent without needing a second hash function.
        """
        digest = hashlib.md5(item).digest()
        offset = 0
        for i in range(self.num_hashes):
            slice_index = i % 8
            if slice_index == 0 and i != 0:
                # Exhausted the current digest; hash it again for more slices.
                digest = hashlib.md5(digest).digest()
                offset = 0
            pos = int.from_bytes(digest[offset:offset + 2], "big") % self.num_bits
            yield pos
            offset += 2

    def add(self, item):
        """Add ``item`` to the filter. ``item`` must be bytes.

        Returns the item, so callers can chain if they want.
        """
        if not isinstance(item, (bytes, bytearray)):
            raise TypeError("item must be bytes")
        for pos in self._positions(item):
            self._bits[pos >> 3] |= 1 << (pos & 7)
        self.count += 1
        return item

    def __contains__(self, item):
        """Return True if ``item`` might be in the filter, False if it is not.

        A False result is certain. A True result has probability at most
        ``error_rate`` of being wrong, provided fewer than ``capacity`` items
        have been added.
        """
        if not isinstance(item, (bytes, bytearray)):
            raise TypeError("item must be bytes")
        for pos in self._positions(item):
            if not (self._bits[pos >> 3] & (1 << (pos & 7))):
                return False
        return True

    def __len__(self):
        """Return the number of items added via :meth:`add`."""
        return self.count

    def __repr__(self):
        return (
            f"BloomFilter(capacity={self.capacity}, "
            f"error_rate={self.error_rate!r}, "
            f"num_bits={self.num_bits}, num_hashes={self.num_hashes}, "
            f"count={self.count})"
        )
