from collections import OrderedDict


class DictQueue:
  """
  A FIFO (First-In-First-Out) dictionary-based queue with maximum length limit.
  Maintains insertion order and automatically evicts oldest elements when full.
  """

  def __init__(self, maxlen):
    if maxlen <= 0:
      raise ValueError("maxlen must be greater than 0")
    self.maxlen = maxlen
    self.data = OrderedDict()

  def put(self, key, value):
    """Add a key-value pair to the queue. Evicts oldest item if full."""
    if key in self.data:
      # Update existing key's value without changing insertion order
      self.data[key] = value
    else:
      # Handle new key with capacity check
      if len(self.data) >= self.maxlen:
        self.data.popitem(last=False)
      self.data[key] = value

  def get(self):
    """Remove and return the oldest (key, value) pair as a tuple."""
    if not self.data:
      raise KeyError("Queue is empty")
    return self.data.popitem(last=False)

  def peek(self):
    """Return (but don't remove) the oldest (key, value) pair as a tuple."""
    if not self.data:
      raise KeyError("Queue is empty")
    key = next(iter(self.data))
    return (key, self.data[key])

  def __contains__(self, key):
    return key in self.data

  def __getitem__(self, key):
    return self.data[key]

  def __len__(self):
    return len(self.data)

  def __repr__(self):
    return f"DictQueue({list(self.data.items())})"


# Usage Example
if __name__ == "__main__":
  q = DictQueue(3)
  q.put('a', 1)
  q.put('b', 2)
  q.put('c', 3)
  print(q)  # Output: DictQueue([('a', 1), ('b', 2), ('c', 3)])

  q.put('d', 4)  # Queue full, evict 'a'
  print(q)  # Output: DictQueue([('b', 2), ('c', 3), ('d', 4)])

  q.put('b', 20)  # Update existing key's value
  print(q)  # Output: DictQueue([('b', 20), ('c', 3), ('d', 4)])

  print(q.get())  # Output: ('b', 20)
  print(q)  # Output: DictQueue([('c', 3), ('d', 4)])
