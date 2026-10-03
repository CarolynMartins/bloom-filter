# bloom_filter

A space-efficient set membership test with a tunable false positive rate. Pure Python, standard library only.

## Usage

```python
from bloom_filter import BloomFilter

bf = BloomFilter(capacity=1000, error_rate=0.01)
bf.add(b"alice@example.com")

if b"alice@example.com" in bf:
    print("maybe present")

if b"stranger@example.com" not in bf:
    print("definitely absent")
```

Items must be `bytes`. Encode strings before adding them; the filter does not guess an encoding.

## Why this exists

A Bloom filter answers "is this item definitely not in the set?" in constant time and a fraction of the memory of a real set. The trade-off is a configurable probability of false positives: it may say an item is present when it is not. It never produces false negatives.

This library targets the case where you want the standard data structure with no third-party dependencies and behaviour you can reason about from the source. It uses MD5 for hashing — not for cryptographic strength, but because it is fast, well-distributed, and present in every Python standard library. The k hash positions are derived by slicing a single MD5 digest into 16-bit integers, re-hashing the digest when k exceeds eight slices.

## Edge cases you will hit

- **Overfilling.** The false positive rate is guaranteed only up to `capacity`. Add more items and the rate rises. The filter does not reject extra items; it is your responsibility to stay within budget.
- **Item type.** Only `bytes` and `bytearray` are accepted. Passing `str` raises `TypeError`. This is deliberate: silent encoding coercion has caused real bugs.
- **`bool` capacity.** `BloomFilter(True)` is rejected. `bool` is a subclass of `int`, but a capacity of `True` is almost always a caller mistake.
- **Very low error rates.** A low `error_rate` relative to `capacity` produces a large number of hash functions. This works but each `add` and `__contains__` call runs proportionally more MD5 invocations.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

