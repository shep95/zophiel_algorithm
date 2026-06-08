"""
code_engine.py — Zophiel Code Generation Engine
================================================
Detects coding requests and returns verified, working code.
Scored against:
  - Human code review standards (readability, correctness, edge cases)
  - AI benchmarks: HumanEval, MBPP, SWE-Bench style checks
  - Big-O complexity awareness
  - Type hints, docstrings, edge case handling
"""
from __future__ import annotations
import re

# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
_CODE_TRIGGERS = re.compile(
    r"\b(write|create|build|implement|code|generate|make|show me|give me)\b.{0,40}"
    r"\b(function|class|algorithm|program|script|method|decorator|query|sql|snippet)\b"
    r"|\b(how (do|to|would) (i|you|we) (implement|write|code|build))\b"
    r"|\b(fibonacci|quicksort|mergesort|binary search|linked list|stack|queue|"
    r"palindrome|factorial|flatten|decorator|traverse|traversal|recursion|sorting|hashing|"
    r"tree|lru cache|two sum|singleton|merge sort)\b",
    re.I,
)

def is_coding_request(query: str) -> bool:
    return bool(_CODE_TRIGGERS.search(query))


# ---------------------------------------------------------------------------
# Verified code library — each entry: (trigger_patterns, language, code, explanation)
# ---------------------------------------------------------------------------
_LIBRARY: list[dict] = [

    # ── Data Structures ─────────────────────────────────────────────────────

    {
        "keys": ["reverse linked list", "reverse a linked list", "linked list reverse"],
        "lang": "python",
        "title": "Reverse a Linked List",
        "complexity": "Time: O(n) | Space: O(1)",
        "code": '''\
from __future__ import annotations
from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, nxt: "Optional[ListNode]" = None):
        self.val = val
        self.next = nxt

def reverse_linked_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """Reverse a singly linked list in-place.

    Uses three-pointer technique: prev trails behind, curr advances,
    next_node saves the forward reference before we break the link.
    """
    prev: Optional[ListNode] = None
    curr = head
    while curr:
        next_node = curr.next   # save forward reference
        curr.next = prev        # reverse the link
        prev = curr             # advance prev
        curr = next_node        # advance curr
    return prev                 # prev is now the new head

# --- Test ---
def to_list(head: Optional[ListNode]) -> list[int]:
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result

def from_list(vals: list[int]) -> Optional[ListNode]:
    if not vals:
        return None
    head = ListNode(vals[0])
    cur = head
    for v in vals[1:]:
        cur.next = ListNode(v)
        cur = cur.next
    return head

assert to_list(reverse_linked_list(from_list([1, 2, 3, 4, 5]))) == [5, 4, 3, 2, 1]
assert to_list(reverse_linked_list(from_list([]))) == []
assert to_list(reverse_linked_list(from_list([1]))) == [1]
print("All tests passed.")
''',
    },

    {
        "keys": ["stack data structure", "implement a stack", "stack class", "python stack"],
        "lang": "python",
        "title": "Stack Data Structure",
        "complexity": "Push/Pop/Peek: O(1) | Space: O(n)",
        "code": '''\
from __future__ import annotations
from typing import Generic, TypeVar, Optional

T = TypeVar("T")

class Stack(Generic[T]):
    """LIFO stack backed by a Python list.

    All core operations are O(1) amortized.
    Raises IndexError on pop/peek of empty stack (fail-fast).
    """

    def __init__(self) -> None:
        self._data: list[T] = []

    def push(self, item: T) -> None:
        self._data.append(item)

    def pop(self) -> T:
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self) -> T:
        if self.is_empty():
            raise IndexError("peek at empty stack")
        return self._data[-1]

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Stack({self._data!r})"

# --- Test ---
s: Stack[int] = Stack()
s.push(1); s.push(2); s.push(3)
assert s.peek() == 3
assert s.pop() == 3
assert len(s) == 2
assert not s.is_empty()
s.pop(); s.pop()
assert s.is_empty()
try:
    s.pop()
    assert False, "should raise"
except IndexError:
    pass
print("All tests passed.")
''',
    },

    {
        "keys": ["queue data structure", "implement a queue", "queue class"],
        "lang": "python",
        "title": "Queue Data Structure",
        "complexity": "Enqueue/Dequeue: O(1) | Space: O(n)",
        "code": '''\
from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")

class Queue(Generic[T]):
    """FIFO queue backed by collections.deque for O(1) enqueue and dequeue."""

    def __init__(self) -> None:
        self._data: deque[T] = deque()

    def enqueue(self, item: T) -> None:
        self._data.append(item)

    def dequeue(self) -> T:
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._data.popleft()

    def front(self) -> T:
        if self.is_empty():
            raise IndexError("front of empty queue")
        return self._data[0]

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def __len__(self) -> int:
        return len(self._data)

# --- Test ---
q: Queue[str] = Queue()
q.enqueue("a"); q.enqueue("b"); q.enqueue("c")
assert q.front() == "a"
assert q.dequeue() == "a"
assert len(q) == 2
print("All tests passed.")
''',
    },

    # ── Search Algorithms ───────────────────────────────────────────────────

    {
        "keys": ["binary search", "binary search algorithm", "implement binary search"],
        "lang": "python",
        "title": "Binary Search",
        "complexity": "Time: O(log n) | Space: O(1)",
        "code": '''\
from typing import TypeVar, Optional

T = TypeVar("T", int, float, str)

def binary_search(arr: list[T], target: T) -> int:
    """Return index of target in sorted arr, or -1 if not found.

    Classic left/right pointer approach. Works on any comparable type.
    Precondition: arr must be sorted in ascending order.
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2   # avoids overflow vs (l+r)//2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# --- Test ---
nums = [1, 3, 5, 7, 9, 11, 13]
assert binary_search(nums, 7)  == 3
assert binary_search(nums, 1)  == 0
assert binary_search(nums, 13) == 6
assert binary_search(nums, 4)  == -1
assert binary_search([], 1)    == -1
print("All tests passed.")
''',
    },

    # ── Sorting Algorithms ──────────────────────────────────────────────────

    {
        "keys": ["quicksort", "quick sort", "implement quicksort"],
        "lang": "python",
        "title": "Quicksort",
        "complexity": "Time: O(n log n) avg, O(n²) worst | Space: O(log n)",
        "code": '''\
def quicksort(arr: list[int]) -> list[int]:
    """In-place quicksort using Lomuto partition scheme.

    Pivot = last element. Average O(n log n); worst case O(n^2) on
    already-sorted input (use randomized pivot to mitigate).
    """
    def _sort(lo: int, hi: int) -> None:
        if lo >= hi:
            return
        pivot_idx = _partition(lo, hi)
        _sort(lo, pivot_idx - 1)
        _sort(pivot_idx + 1, hi)

    def _partition(lo: int, hi: int) -> int:
        pivot = arr[hi]
        i = lo - 1
        for j in range(lo, hi):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
        return i + 1

    _sort(0, len(arr) - 1)
    return arr

# --- Test ---
assert quicksort([3, 6, 8, 10, 1, 2, 1]) == [1, 1, 2, 3, 6, 8, 10]
assert quicksort([])  == []
assert quicksort([1]) == [1]
assert quicksort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]
print("All tests passed.")
''',
    },

    {
        "keys": ["mergesort", "merge sort", "implement mergesort"],
        "lang": "python",
        "title": "Merge Sort",
        "complexity": "Time: O(n log n) | Space: O(n)",
        "code": '''\
def mergesort(arr: list[int]) -> list[int]:
    """Stable, divide-and-conquer sort. Guaranteed O(n log n) always.

    Creates new sublists — not in-place. Use when stability matters
    or when worst-case guarantees matter more than memory.
    """
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left  = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    return _merge(left, right)

def _merge(left: list[int], right: list[int]) -> list[int]:
    result: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# --- Test ---
assert mergesort([5, 2, 4, 6, 1, 3]) == [1, 2, 3, 4, 5, 6]
assert mergesort([]) == []
assert mergesort([1, 1, 1]) == [1, 1, 1]
print("All tests passed.")
''',
    },

    # ── String Algorithms ───────────────────────────────────────────────────

    {
        "keys": ["palindrome", "check palindrome", "is palindrome", "string palindrome"],
        "lang": "python",
        "title": "Palindrome Check",
        "complexity": "Time: O(n) | Space: O(1)",
        "code": '''\
import re

def is_palindrome(s: str, alphanumeric_only: bool = True) -> bool:
    """Check if s reads the same forwards and backwards.

    alphanumeric_only=True (default): ignores spaces, punctuation, case.
    alphanumeric_only=False: exact character match.
    """
    if alphanumeric_only:
        s = re.sub(r"[^a-zA-Z0-9]", "", s).lower()
    return s == s[::-1]

# --- Test ---
assert is_palindrome("racecar")           == True
assert is_palindrome("A man a plan a canal Panama") == True
assert is_palindrome("hello")             == False
assert is_palindrome("")                  == True
assert is_palindrome("Was it a car or a cat I saw?") == True
assert is_palindrome("hello", alphanumeric_only=False) == False
print("All tests passed.")
''',
    },

    # ── Dynamic Programming ─────────────────────────────────────────────────

    {
        "keys": ["fibonacci", "fibonacci sequence", "fibonacci number", "fib sequence"],
        "lang": "python",
        "title": "Fibonacci Sequence",
        "complexity": "Iterative: O(n) time O(1) space | Memoized: O(n) time O(n) space",
        "code": '''\
from functools import lru_cache

def fibonacci_iterative(n: int) -> int:
    """Return the nth Fibonacci number (0-indexed). O(n) time, O(1) space."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

@lru_cache(maxsize=None)
def fibonacci_recursive(n: int) -> int:
    """Memoized recursive variant. Clean but uses O(n) stack space."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def fibonacci_sequence(count: int) -> list[int]:
    """Return the first `count` Fibonacci numbers."""
    return [fibonacci_iterative(i) for i in range(count)]

# --- Test ---
assert fibonacci_iterative(0)  == 0
assert fibonacci_iterative(1)  == 1
assert fibonacci_iterative(10) == 55
assert fibonacci_iterative(20) == 6765
assert fibonacci_sequence(8)   == [0, 1, 1, 2, 3, 5, 8, 13]
assert fibonacci_recursive(10) == 55
print("All tests passed.")
''',
    },

    {
        "keys": ["factorial", "compute factorial", "recursive factorial"],
        "lang": "python",
        "title": "Factorial",
        "complexity": "Iterative: O(n) time O(1) space",
        "code": '''\
def factorial(n: int) -> int:
    """Return n! for non-negative integer n.

    Iterative to avoid Python recursion limit on large n.
    """
    if n < 0:
        raise ValueError("factorial undefined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# --- Test ---
assert factorial(0)  == 1
assert factorial(1)  == 1
assert factorial(5)  == 120
assert factorial(10) == 3628800
print("All tests passed.")
''',
    },

    # ── Functional / Higher-Order ────────────────────────────────────────────

    {
        "keys": ["flatten", "flatten a nested list", "flatten nested list", "flatten list"],
        "lang": "python",
        "title": "Flatten Nested List",
        "complexity": "Time: O(n) | Space: O(depth) recursion stack",
        "code": '''\
from typing import Any

def flatten(nested: list[Any]) -> list[Any]:
    """Recursively flatten an arbitrarily deep nested list.

    flatten([1, [2, [3, 4]], 5]) -> [1, 2, 3, 4, 5]
    Non-list items are kept as-is.
    """
    result: list[Any] = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

# Iterative variant (avoids recursion limit for very deep nesting)
def flatten_iterative(nested: list[Any]) -> list[Any]:
    result: list[Any] = []
    stack = list(reversed(nested))   # push in reverse so first item pops first
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))   # push sublist items in reverse order
        else:
            result.append(item)
    return result

# --- Test ---
assert flatten([1, [2, [3, 4]], 5])           == [1, 2, 3, 4, 5]
assert flatten([])                             == []
assert flatten([[1, 2], [3, [4, [5]]]])       == [1, 2, 3, 4, 5]
assert flatten_iterative([1, [2, [3, 4]]])    == [1, 2, 3, 4]
assert flatten_iterative([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]
print("All tests passed.")
''',
    },

    {
        "keys": ["timing decorator", "time decorator", "decorator times", "execution time decorator", "decorator that times"],
        "lang": "python",
        "title": "Timing Decorator",
        "complexity": "Wrapper overhead: O(1)",
        "code": '''\
import time
import functools
from typing import Callable, Any

def timer(func: Callable) -> Callable:
    """Decorator that prints the execution time of the wrapped function.

    Preserves the original function name and docstring via functools.wraps.
    Returns the function result unchanged.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start  = time.perf_counter()
        result = func(*args, **kwargs)
        end    = time.perf_counter()
        print(f"{func.__name__!r} took {(end - start) * 1000:.3f} ms")
        return result
    return wrapper

# Parametrized variant: @timer(unit="s")
def timer_unit(unit: str = "ms") -> Callable:
    divisor = 1 if unit == "ms" else 1000
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start  = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000 / divisor
            print(f"{func.__name__!r} took {elapsed:.3f} {unit}")
            return result
        return wrapper
    return decorator

# --- Test ---
@timer
def slow_add(a: int, b: int) -> int:
    time.sleep(0.01)
    return a + b

result = slow_add(3, 4)
assert result == 7
assert slow_add.__name__ == "slow_add"   # wraps preserved name
print("All tests passed.")
''',
    },

    # ── Tree Algorithms ─────────────────────────────────────────────────────

    {
        "keys": ["tree traversal", "binary tree traversal", "recursive tree traversal",
                 "inorder traversal", "preorder traversal", "postorder traversal"],
        "lang": "python",
        "title": "Binary Tree Traversal (In/Pre/Post order)",
        "complexity": "Time: O(n) | Space: O(h) where h = tree height",
        "code": '''\
from __future__ import annotations
from typing import Optional

class TreeNode:
    def __init__(self, val: int = 0,
                 left: "Optional[TreeNode]" = None,
                 right: "Optional[TreeNode]" = None):
        self.val   = val
        self.left  = left
        self.right = right

def inorder(root: Optional[TreeNode]) -> list[int]:
    """Left -> Root -> Right. Gives sorted order for a BST."""
    result: list[int] = []
    def _visit(node: Optional[TreeNode]) -> None:
        if not node:
            return
        _visit(node.left)
        result.append(node.val)
        _visit(node.right)
    _visit(root)
    return result

def preorder(root: Optional[TreeNode]) -> list[int]:
    """Root -> Left -> Right. Useful for tree serialization."""
    result: list[int] = []
    def _visit(node: Optional[TreeNode]) -> None:
        if not node:
            return
        result.append(node.val)
        _visit(node.left)
        _visit(node.right)
    _visit(root)
    return result

def postorder(root: Optional[TreeNode]) -> list[int]:
    """Left -> Right -> Root. Useful for deletion / evaluation."""
    result: list[int] = []
    def _visit(node: Optional[TreeNode]) -> None:
        if not node:
            return
        _visit(node.left)
        _visit(node.right)
        result.append(node.val)
    _visit(root)
    return result

# Build tree:   4
#              / \\
#             2   6
#            /\\ /\\
#           1 3 5 7
root = TreeNode(4,
    TreeNode(2, TreeNode(1), TreeNode(3)),
    TreeNode(6, TreeNode(5), TreeNode(7)),
)
assert inorder(root)   == [1, 2, 3, 4, 5, 6, 7]
assert preorder(root)  == [4, 2, 1, 3, 6, 5, 7]
assert postorder(root) == [1, 3, 2, 5, 7, 6, 4]
assert inorder(None)   == []
print("All tests passed.")
''',
    },

    # ── SQL ─────────────────────────────────────────────────────────────────

    {
        "keys": ["duplicate rows", "find duplicates sql", "sql duplicate", "duplicate sql"],
        "lang": "sql",
        "title": "Find Duplicate Rows in SQL",
        "complexity": "Depends on indexes; GROUP BY is O(n log n)",
        "code": '''\
-- Method 1: GROUP BY + HAVING (most portable)
-- Returns the duplicate value and how many times it appears
SELECT email, COUNT(*) AS occurrences
FROM users
GROUP BY email
HAVING COUNT(*) > 1
ORDER BY occurrences DESC;

-- Method 2: Return the full duplicate rows (including all columns)
-- Using a self-join on the duplicate key
SELECT u.*
FROM users u
INNER JOIN (
    SELECT email
    FROM users
    GROUP BY email
    HAVING COUNT(*) > 1
) dups ON u.email = dups.email
ORDER BY u.email;

-- Method 3: ROW_NUMBER() window function (PostgreSQL / SQL Server / MySQL 8+)
-- Lets you identify and delete duplicates while keeping one copy
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY email ORDER BY id) AS rn
    FROM users
)
SELECT * FROM ranked WHERE rn > 1;   -- rows to delete

-- To DELETE duplicates keeping the lowest id:
-- DELETE FROM ranked WHERE rn > 1;
''',
    },

    # ── OOP / Design Patterns ────────────────────────────────────────────────

    {
        "keys": ["singleton pattern", "singleton class", "implement singleton"],
        "lang": "python",
        "title": "Singleton Pattern",
        "complexity": "O(1) instance lookup",
        "code": '''\
from threading import Lock
from typing import Optional, Any

class Singleton:
    """Thread-safe Singleton using double-checked locking.

    Only one instance is ever created; subsequent calls return the same object.
    """
    _instance: Optional["Singleton"] = None
    _lock: Lock = Lock()

    def __new__(cls) -> "Singleton":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:   # second check inside lock
                    cls._instance = super().__new__(cls)
        return cls._instance

# --- Test ---
a = Singleton()
b = Singleton()
assert a is b, "Should be the same instance"
print(f"Same instance: {a is b}")
print("All tests passed.")
''',
    },

    {
        "keys": ["two sum", "twosum", "two sum problem", "find two numbers that add"],
        "lang": "python",
        "title": "Two Sum (Hash Map)",
        "complexity": "Time: O(n) | Space: O(n)",
        "code": '''\
def two_sum(nums: list[int], target: int) -> list[int]:
    """Return indices of two numbers that add up to target.

    Uses a hash map for O(n) time. Assumes exactly one solution exists.
    """
    seen: dict[int, int] = {}   # value -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []   # no solution found

# --- Test ---
assert two_sum([2, 7, 11, 15], 9)  == [0, 1]
assert two_sum([3, 2, 4], 6)       == [1, 2]
assert two_sum([3, 3], 6)          == [0, 1]
print("All tests passed.")
''',
    },

    {
        "keys": ["lru cache", "implement lru", "lru cache class", "least recently used"],
        "lang": "python",
        "title": "LRU Cache",
        "complexity": "Get/Put: O(1) | Space: O(capacity)",
        "code": '''\
from collections import OrderedDict

class LRUCache:
    """Least Recently Used cache with O(1) get and put.

    Uses OrderedDict to maintain insertion order and move items to the end
    on access (end = most recently used, front = least recently used).
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._cache: OrderedDict[int, int] = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self._cache:
            return -1
        self._cache.move_to_end(key)   # mark as recently used
        return self._cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)  # evict least recently used

# --- Test ---
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1)  == 1       # returns 1, marks 1 as recently used
cache.put(3, 3)                  # evicts key 2
assert cache.get(2)  == -1      # evicted
cache.put(4, 4)                  # evicts key 1
assert cache.get(1)  == -1      # evicted
assert cache.get(3)  == 3
assert cache.get(4)  == 4
print("All tests passed.")
''',
    },
]


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------
def _match(query: str) -> dict | None:
    q = query.lower().strip()
    # exact key match first
    for entry in _LIBRARY:
        for key in entry["keys"]:
            if key in q:
                return entry
    # fuzzy: score by word overlap
    q_words = set(re.findall(r"[a-z]+", q))
    best_score = 0
    best_entry = None
    for entry in _LIBRARY:
        for key in entry["keys"]:
            k_words = set(re.findall(r"[a-z]+", key))
            overlap  = len(q_words & k_words)
            if overlap > best_score and overlap >= 2:
                best_score = overlap
                best_entry = entry
    return best_entry


def generate_code(query: str) -> str | None:
    """Return a formatted code answer for coding requests, or None."""
    if not is_coding_request(query):
        return None
    entry = _match(query)
    if not entry:
        return None
    lang  = entry["lang"]
    title = entry["title"]
    comp  = entry.get("complexity", "")
    code  = entry["code"].strip()
    lines = [
        f"## {title}",
        f"**Complexity:** {comp}" if comp else "",
        f"```{lang}",
        code,
        "```",
    ]
    return "\n".join(l for l in lines if l)
