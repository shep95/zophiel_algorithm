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
    r"tree|lru cache|two sum|singleton|merge sort|context manager|generator|prime number|"
    r"password|bfs|dfs|caesar cipher|inheritance|rate limiter|rate limiting|matrix multipl|"
    r"matrix multiply|csv pars|async fetch|async http|aiohttp|"
    # Data Structures
    r"doubly linked|min heap|max heap|binary heap|trie|prefix tree|hash table|hash map|"
    r"circular buffer|ring buffer|deque|double.ended queue|"
    # Sorting
    r"bubble sort|insertion sort|selection sort|heap sort|counting sort|radix sort|timsort|"
    # Search
    r"linear search|jump search|interpolation search|"
    # Graph
    r"dijkstra|topological sort|union find|disjoint set|kruskal|minimum spanning|cycle detect|"
    # Dynamic Programming
    r"knapsack|longest common subsequence|lcs|longest increasing subsequence|lis|"
    r"coin change|edit distance|levenshtein|kadane|max subarray|"
    # Strings
    r"anagram|run.length encoding|rle compress|kmp|knuth morris|string permut|word count|word freq|"
    # Math
    r"gcd|lcm|greatest common|least common|fast power|exponent|prime factor|armstrong number|"
    r"number to binary|number to hex|base convert|"
    # Design Patterns
    r"factory pattern|observer pattern|strategy pattern|builder pattern|command pattern|"
    r"decorator pattern|"
    # Functional
    r"map filter reduce|memoiz|currying|function composition|partial application|"
    # File I/O
    r"json read|json write|directory walk|file copy|log pars|"
    # Networking
    r"email valid|url pars|http server|"
    # SQL
    r"crud|sql aggregat|window function|join type|"
    # Security
    r"base64|sha256|xor cipher|hmac|"
    # Data Processing
    r"mean median mode|moving average|data normaliz|"
    # Concurrency
    r"thread pool|producer consumer|semaphore|"
    # Recursion/Backtracking
    r"tower of hanoi|hanoi|permutation|combination|power set|n.queens|queens problem|"
    # Interview Classics
    r"valid parentheses|balanced brackets|merge intervals|product except self|"
    r"sliding window max|rotate array|dutch national flag|three way partition)\b"
    r"|\b(write|build|implement|create).{0,30}(rate limit|matrix|heap|trie|graph|cache|"
    r"blockchain|neural network|url short|sudoku|event bus|scheduler|hash ring)\b"
    r"|\b(blockchain|proof of work|neural network|url shortener|sudoku|pub.?sub|"
    r"event bus|job scheduler|consistent hash|hash ring|pubsub)\b",
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

    # ── Context Manager ─────────────────────────────────────────────────────

    {
        "keys": ["context manager", "file handling context", "with statement", "__enter__ __exit__"],
        "lang": "python",
        "title": "Context Manager (File Handling)",
        "complexity": "O(1) open/close overhead",
        "code": '''\
from __future__ import annotations
import os
from types import TracebackType
from typing import Optional, Type, IO, Any

class ManagedFile:
    """Context manager that opens a file and guarantees it is closed.

    Handles exceptions inside the with-block gracefully.
    Supports both reading and writing modes.
    """

    def __init__(self, path: str, mode: str = "r", encoding: str = "utf-8") -> None:
        self.path     = path
        self.mode     = mode
        self.encoding = encoding
        self._file: Optional[IO[Any]] = None

    def __enter__(self) -> IO[Any]:
        self._file = open(self.path, self.mode, encoding=self.encoding)
        return self._file

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val:  Optional[BaseException],
        exc_tb:   Optional[TracebackType],
    ) -> bool:
        if self._file:
            self._file.close()
        return False   # re-raise any exception — do not suppress

# Equivalent using @contextmanager decorator (simpler for one-offs)
from contextlib import contextmanager

@contextmanager
def open_file(path: str, mode: str = "r"):
    """Minimal context manager using generator protocol."""
    f = open(path, mode, encoding="utf-8")
    try:
        yield f
    finally:
        f.close()

# --- Test ---
import tempfile, os

with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
    tmp_path = tmp.name
    tmp.write("hello world")

with ManagedFile(tmp_path, "r") as f:
    content = f.read()
assert content == "hello world"

with open_file(tmp_path, "r") as f:
    content2 = f.read()
assert content2 == "hello world"

os.unlink(tmp_path)
print("All tests passed.")
''',
    },

    # ── Generator ───────────────────────────────────────────────────────────

    {
        "keys": ["generator prime", "yield prime", "prime numbers generator", "prime number generator",
                 "generator that yields prime", "yields prime numbers"],
        "lang": "python",
        "title": "Prime Number Generator",
        "complexity": "Sieve: O(n log log n) | Trial division per number: O(sqrt(n))",
        "code": '''\
from typing import Generator, Iterator

def is_prime(n: int) -> bool:
    """Trial division primality test. O(sqrt(n))."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def prime_generator(limit: int = None) -> Generator[int, None, None]:
    """Yield prime numbers indefinitely (or up to limit).

    Uses trial division — good for sparse primes; use Sieve for bulk.
    """
    n = 2
    count = 0
    while limit is None or count < limit:
        if is_prime(n):
            yield n
            count += 1
        n += 1

def sieve_of_eratosthenes(n: int) -> list[int]:
    """Return all primes <= n using the Sieve of Eratosthenes. O(n log log n)."""
    if n < 2:
        return []
    composite = bytearray(n + 1)   # 0 = prime, 1 = composite
    composite[0] = composite[1] = 1
    for i in range(2, int(n ** 0.5) + 1):
        if not composite[i]:
            composite[i * i::i] = b"\\x01" * len(composite[i * i::i])
    return [i for i in range(2, n + 1) if not composite[i]]

# --- Test ---
first_10 = list(prime_generator(10))
assert first_10 == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
assert sieve_of_eratosthenes(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
assert not is_prime(1)
assert is_prime(97)
print("All tests passed.")
''',
    },

    # ── Security — Password Validator ────────────────────────────────────────

    {
        "keys": ["password strength", "password validator", "validate password",
                 "password checker", "strong password"],
        "lang": "python",
        "title": "Password Strength Validator",
        "complexity": "O(n) where n = password length",
        "code": '''\
import re
from dataclasses import dataclass, field

@dataclass
class PasswordResult:
    score: int          # 0-100
    strength: str       # Weak / Fair / Strong / Very Strong
    passed: bool
    failures: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

def validate_password(password: str, min_length: int = 8) -> PasswordResult:
    """Evaluate password strength against common security criteria.

    Scoring:
      Length >= 8:        +20   Length >= 12:    +10   Length >= 16: +10
      Uppercase letter:   +15   Lowercase letter: +15
      Digit:              +15   Special char:     +15
    """
    failures: list[str] = []
    suggestions: list[str] = []
    score = 0

    if len(password) < min_length:
        failures.append(f"Must be at least {min_length} characters")
    else:
        score += 20
        if len(password) >= 12: score += 10
        if len(password) >= 16: score += 10

    has_upper = bool(re.search(r"[A-Z]", password))
    has_lower = bool(re.search(r"[a-z]", password))
    has_digit = bool(re.search(r"\\d", password))
    has_special = bool(re.search(r"[!@#$%^&*()_+\\-=\\[\\]{};\':\\"\\\\|,.<>\\/?]", password))

    if has_upper:  score += 15
    else: failures.append("Add uppercase letters"); suggestions.append("Include A-Z")

    if has_lower:  score += 15
    else: failures.append("Add lowercase letters"); suggestions.append("Include a-z")

    if has_digit:  score += 15
    else: failures.append("Add numbers"); suggestions.append("Include 0-9")

    if has_special: score += 15
    else: suggestions.append("Add special chars like !@#$ for extra strength")

    # Common pattern penalties
    if re.search(r"(.)\\1{2,}", password):      # 3+ repeated chars
        score = max(0, score - 10)
        suggestions.append("Avoid repeated characters (aaa, 111)")
    if re.search(r"(012|123|234|345|456|567|678|789|890|abc|bcd|qwerty|password)", password.lower()):
        score = max(0, score - 15)
        failures.append("Avoid sequential or common patterns")

    score = min(100, score)
    if score < 40:   strength = "Weak"
    elif score < 60: strength = "Fair"
    elif score < 80: strength = "Strong"
    else:            strength = "Very Strong"

    return PasswordResult(
        score=score,
        strength=strength,
        passed=len(failures) == 0 and score >= 60,
        failures=failures,
        suggestions=suggestions,
    )

# --- Test ---
assert validate_password("abc").passed == False
assert validate_password("abc").strength == "Weak"
result = validate_password("MyP@ssw0rd!")
assert result.score >= 60
assert result.passed == True
result2 = validate_password("C0rr3ct#HorseBatteryStaple!")
assert result2.strength in ("Strong", "Very Strong")
print("All tests passed.")
''',
    },

    # ── Graph Algorithms ─────────────────────────────────────────────────────

    {
        "keys": ["graph bfs", "breadth first search", "bfs algorithm", "bfs graph",
                 "graph breadth", "bfs traversal"],
        "lang": "python",
        "title": "Graph BFS and DFS",
        "complexity": "BFS/DFS: O(V + E) | Space: O(V)",
        "code": '''\
from collections import deque
from typing import Optional

# Graph represented as adjacency list
Graph = dict[str, list[str]]

def bfs(graph: Graph, start: str) -> list[str]:
    """Breadth-First Search — explores layer by layer.

    Guarantees shortest path (hop count) in unweighted graphs.
    """
    visited: set[str] = {start}
    queue: deque[str] = deque([start])
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return order

def dfs(graph: Graph, start: str) -> list[str]:
    """Depth-First Search — explores as deep as possible before backtracking."""
    visited: set[str] = set()
    order: list[str] = []

    def _visit(node: str) -> None:
        visited.add(node)
        order.append(node)
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                _visit(neighbour)

    _visit(start)
    return order

def shortest_path(graph: Graph, start: str, end: str) -> Optional[list[str]]:
    """BFS-based shortest path. Returns node list or None if unreachable."""
    if start == end:
        return [start]
    visited = {start}
    queue: deque[list[str]] = deque([[start]])

    while queue:
        path = queue.popleft()
        for neighbour in graph.get(path[-1], []):
            if neighbour not in visited:
                new_path = path + [neighbour]
                if neighbour == end:
                    return new_path
                visited.add(neighbour)
                queue.append(new_path)
    return None

# --- Test ---
g: Graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}
assert bfs(g, "A") == ["A", "B", "C", "D", "E", "F"]
assert dfs(g, "A") == ["A", "B", "D", "E", "F", "C"]
assert shortest_path(g, "A", "F") == ["A", "C", "F"]
assert shortest_path(g, "A", "A") == ["A"]
assert shortest_path(g, "D", "F") == ["D", "B", "E", "F"]
print("All tests passed.")
''',
    },

    # ── Cryptography ────────────────────────────────────────────────────────

    {
        "keys": ["caesar cipher", "caesar encrypt", "caesar decrypt", "rot cipher", "caesar code"],
        "lang": "python",
        "title": "Caesar Cipher (Encrypt & Decrypt)",
        "complexity": "O(n) where n = message length",
        "code": '''\
def caesar_encrypt(text: str, shift: int) -> str:
    """Shift each letter by `shift` positions (wraps at 26). Non-letters unchanged."""
    result: list[str] = []
    shift = shift % 26   # normalise — handles negative shifts too
    for ch in text:
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            result.append(ch)
    return "".join(result)

def caesar_decrypt(text: str, shift: int) -> str:
    """Decrypt by shifting in the opposite direction."""
    return caesar_encrypt(text, -shift)

def caesar_brute_force(ciphertext: str) -> list[tuple[int, str]]:
    """Return all 26 possible decryptions for manual inspection."""
    return [(s, caesar_decrypt(ciphertext, s)) for s in range(26)]

# --- Test ---
assert caesar_encrypt("Hello, World!", 3) == "Khoor, Zruog!"
assert caesar_decrypt("Khoor, Zruog!", 3) == "Hello, World!"
assert caesar_encrypt("xyz", 3)           == "abc"
assert caesar_encrypt("ABC", 1)           == "BCD"
assert caesar_encrypt("Hello!", 0)        == "Hello!"
assert caesar_decrypt(caesar_encrypt("Secret message", 13), 13) == "Secret message"
# ROT13 is its own inverse
assert caesar_encrypt(caesar_encrypt("Python", 13), 13) == "Python"
print("All tests passed.")
''',
    },

    # ── OOP — Inheritance ────────────────────────────────────────────────────

    {
        "keys": ["inheritance animal", "class inheritance animal", "animal class",
                 "python inheritance", "oop inheritance", "base class derived class"],
        "lang": "python",
        "title": "OOP Inheritance — Animal Hierarchy",
        "complexity": "Method dispatch: O(1)",
        "code": '''\
from __future__ import annotations
from abc import ABC, abstractmethod

class Animal(ABC):
    """Abstract base class — defines the interface every animal must implement."""

    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age  = age

    @abstractmethod
    def speak(self) -> str:
        """Each animal must define its own sound."""
        ...

    @abstractmethod
    def move(self) -> str:
        """Each animal must define how it moves."""
        ...

    def describe(self) -> str:
        return f"{self.name} (age {self.age}): says '{self.speak()}', moves by {self.move()}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, age={self.age})"

class Dog(Animal):
    def speak(self) -> str: return "Woof!"
    def move(self)  -> str: return "running"
    def fetch(self, item: str) -> str: return f"{self.name} fetches the {item}!"

class Cat(Animal):
    def speak(self) -> str: return "Meow!"
    def move(self)  -> str: return "prowling"
    def purr(self) -> str: return f"{self.name} purrs..."

class Bird(Animal):
    def __init__(self, name: str, age: int, can_fly: bool = True) -> None:
        super().__init__(name, age)
        self.can_fly = can_fly

    def speak(self) -> str: return "Tweet!"
    def move(self)  -> str: return "flying" if self.can_fly else "walking"

# --- Test ---
animals: list[Animal] = [
    Dog("Rex", 3),
    Cat("Whiskers", 5),
    Bird("Tweety", 1),
    Bird("Penny the Penguin", 2, can_fly=False),
]
assert animals[0].speak()  == "Woof!"
assert animals[1].speak()  == "Meow!"
assert animals[2].move()   == "flying"
assert animals[3].move()   == "walking"
assert isinstance(animals[0], Animal)
assert isinstance(animals[0], Dog)

# Polymorphism
sounds = [a.speak() for a in animals]
assert sounds == ["Woof!", "Meow!", "Tweet!", "Tweet!"]

try:
    Animal("test", 1)  # type: ignore
    assert False, "Should not instantiate abstract class"
except TypeError:
    pass

print("All tests passed.")
''',
    },

    # ── Rate Limiter ─────────────────────────────────────────────────────────

    {
        "keys": ["rate limiter", "rate limiting", "throttle requests", "token bucket",
                 "sliding window rate"],
        "lang": "python",
        "title": "Rate Limiter (Token Bucket + Sliding Window)",
        "complexity": "O(1) per request check",
        "code": '''\
import time
from collections import deque
from threading import Lock

class TokenBucketRateLimiter:
    """Token bucket algorithm — allows burst up to `capacity`, refills at `rate` per second.

    Thread-safe via Lock. Good for APIs where short bursts are acceptable.
    """

    def __init__(self, capacity: int, rate: float) -> None:
        self.capacity = capacity
        self.rate     = rate          # tokens added per second
        self._tokens  = float(capacity)
        self._last    = time.monotonic()
        self._lock    = Lock()

    def allow(self) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last = now
            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False

class SlidingWindowRateLimiter:
    """Sliding window log — exact per-second request count.

    Keeps timestamps of recent requests; evicts those outside the window.
    """

    def __init__(self, max_requests: int, window_seconds: float = 1.0) -> None:
        self.max_requests = max_requests
        self.window       = window_seconds
        self._log: deque[float] = deque()
        self._lock = Lock()

    def allow(self) -> bool:
        with self._lock:
            now = time.monotonic()
            cutoff = now - self.window
            while self._log and self._log[0] <= cutoff:
                self._log.popleft()
            if len(self._log) < self.max_requests:
                self._log.append(now)
                return True
            return False

# --- Test ---
tb = TokenBucketRateLimiter(capacity=5, rate=1.0)
results = [tb.allow() for _ in range(6)]
assert results[:5] == [True] * 5
assert results[5]  == False      # bucket empty

sw = SlidingWindowRateLimiter(max_requests=3, window_seconds=1.0)
assert sw.allow() == True
assert sw.allow() == True
assert sw.allow() == True
assert sw.allow() == False       # 4th request denied within window
time.sleep(1.05)
assert sw.allow() == True        # window has slid, now allowed again

print("All tests passed.")
''',
    },

    # ── Matrix Multiplication ────────────────────────────────────────────────

    {
        "keys": ["matrix multiplication", "matrix multiply", "multiply matrices",
                 "matrix product", "matmul without numpy"],
        "lang": "python",
        "title": "Matrix Multiplication (No NumPy)",
        "complexity": "Naive: O(n³) | Strassen: O(n^2.807)",
        "code": '''\
from typing import TypeAlias

Matrix: TypeAlias = list[list[float]]

def matmul(A: Matrix, B: Matrix) -> Matrix:
    """Multiply two matrices A (m x k) and B (k x n). Returns m x n matrix.

    Raises ValueError if dimensions are incompatible.
    """
    m, k  = len(A), len(A[0])
    k2, n = len(B), len(B[0])
    if k != k2:
        raise ValueError(f"Incompatible dimensions: ({m}x{k}) @ ({k2}x{n})")
    result: Matrix = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            result[i][j] = sum(A[i][p] * B[p][j] for p in range(k))
    return result

def transpose(A: Matrix) -> Matrix:
    """Transpose an m x n matrix to n x m."""
    if not A:
        return []
    return [[A[r][c] for r in range(len(A))] for c in range(len(A[0]))]

def identity(n: int) -> Matrix:
    """Return the n x n identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

# --- Test ---
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
result = matmul(A, B)
assert result == [[19.0, 22.0], [43.0, 50.0]]

# Identity matrix property: A @ I = A
I = identity(2)
assert matmul(A, I) == [[float(x) for x in row] for row in A]

# Transpose
assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]

# Non-square
C = [[1, 2, 3]]       # 1x3
D = [[4], [5], [6]]   # 3x1
assert matmul(C, D)   == [[32.0]]

try:
    matmul([[1, 2]], [[1, 2]])
    assert False
except ValueError:
    pass

print("All tests passed.")
''',
    },

    # ── CSV Parser ───────────────────────────────────────────────────────────

    {
        "keys": ["csv parser", "parse csv", "csv file manually", "read csv", "csv parsing"],
        "lang": "python",
        "title": "CSV Parser (Manual — No csv module)",
        "complexity": "O(n*m) where n=rows, m=cols",
        "code": '''\
from typing import Iterator

def parse_csv(text: str, delimiter: str = ",", has_header: bool = True) -> list[dict]:
    """Parse CSV text into a list of dicts (when has_header=True) or lists.

    Handles:
      - Quoted fields (commas inside quotes)
      - Escaped quotes (\\"\\")
      - Leading/trailing whitespace in unquoted fields
    """
    def _parse_line(line: str) -> list[str]:
        fields: list[str] = []
        current = ""
        in_quotes = False
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "\\"" and in_quotes and i + 1 < len(line) and line[i + 1] == "\\"":
                current += "\\""; i += 2; continue   # escaped quote
            if ch == "\\"":
                in_quotes = not in_quotes
            elif ch == delimiter and not in_quotes:
                fields.append(current.strip() if not in_quotes else current)
                current = ""
            else:
                current += ch
            i += 1
        fields.append(current.strip())
        return fields

    lines = [l for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return []

    if has_header:
        headers = _parse_line(lines[0])
        return [
            dict(zip(headers, _parse_line(line)))
            for line in lines[1:]
        ]
    return [_parse_line(line) for line in lines]  # type: ignore

# --- Test ---
csv_text = """name,age,city
Alice,30,New York
Bob,25,"Los Angeles"
Charlie,35,"Austin, TX"
"""
rows = parse_csv(csv_text)
assert len(rows) == 3
assert rows[0] == {"name": "Alice", "age": "30", "city": "New York"}
assert rows[2]["city"] == "Austin, TX"   # comma inside quoted field

# No-header mode
raw = parse_csv("1,2,3\\n4,5,6", has_header=False)
assert raw == [["1", "2", "3"], ["4", "5", "6"]]

print("All tests passed.")
''',
    },

    # ── Doubly Linked List ───────────────────────────────────────────────────

    {
        "keys": ["doubly linked list", "doubly linked", "double linked list"],
        "lang": "python",
        "title": "Doubly Linked List",
        "complexity": "Insert/Delete at head or tail: O(1) | Search: O(n)",
        "code": '''\
from __future__ import annotations
from typing import Optional, Generic, TypeVar

T = TypeVar("T")

class DNode(Generic[T]):
    def __init__(self, val: T) -> None:
        self.val  = val
        self.prev: Optional[DNode[T]] = None
        self.next: Optional[DNode[T]] = None

class DoublyLinkedList(Generic[T]):
    """Doubly linked list with O(1) head/tail insert and delete."""

    def __init__(self) -> None:
        self.head: Optional[DNode[T]] = None
        self.tail: Optional[DNode[T]] = None
        self._size = 0

    def append(self, val: T) -> None:
        node = DNode(val)
        if self.tail:
            self.tail.next = node
            node.prev = self.tail
            self.tail = node
        else:
            self.head = self.tail = node
        self._size += 1

    def prepend(self, val: T) -> None:
        node = DNode(val)
        if self.head:
            node.next = self.head
            self.head.prev = node
            self.head = node
        else:
            self.head = self.tail = node
        self._size += 1

    def delete(self, val: T) -> bool:
        cur = self.head
        while cur:
            if cur.val == val:
                if cur.prev: cur.prev.next = cur.next
                else:        self.head = cur.next
                if cur.next: cur.next.prev = cur.prev
                else:        self.tail = cur.prev
                self._size -= 1
                return True
            cur = cur.next
        return False

    def to_list(self) -> list[T]:
        result, cur = [], self.head
        while cur:
            result.append(cur.val)
            cur = cur.next
        return result

    def __len__(self) -> int:
        return self._size

# --- Test ---
dll: DoublyLinkedList[int] = DoublyLinkedList()
dll.append(1); dll.append(2); dll.append(3)
dll.prepend(0)
assert dll.to_list() == [0, 1, 2, 3]
dll.delete(2)
assert dll.to_list() == [0, 1, 3]
assert len(dll) == 3
print("All tests passed.")
''',
    },

    # ── Min Heap ─────────────────────────────────────────────────────────────

    {
        "keys": ["min heap", "max heap", "binary heap", "heap data structure", "implement heap",
                 "priority queue heap"],
        "lang": "python",
        "title": "Min Heap and Max Heap",
        "complexity": "Push/Pop: O(log n) | Peek: O(1) | Build: O(n)",
        "code": '''\
import heapq
from typing import TypeVar, Generic

T = TypeVar("T", int, float)

class MinHeap:
    """Min-heap backed by Python\'s heapq. Root is always the smallest element."""

    def __init__(self) -> None:
        self._data: list[int] = []

    def push(self, val: int) -> None:
        heapq.heappush(self._data, val)

    def pop(self) -> int:
        return heapq.heappop(self._data)

    def peek(self) -> int:
        return self._data[0]

    def __len__(self) -> int:
        return len(self._data)

class MaxHeap:
    """Max-heap by negating values before storing in Python\'s min-heap."""

    def __init__(self) -> None:
        self._data: list[int] = []

    def push(self, val: int) -> None:
        heapq.heappush(self._data, -val)

    def pop(self) -> int:
        return -heapq.heappop(self._data)

    def peek(self) -> int:
        return -self._data[0]

    def __len__(self) -> int:
        return len(self._data)

def kth_smallest(nums: list[int], k: int) -> int:
    """Return the kth smallest element using a max-heap of size k. O(n log k)."""
    heap = MaxHeap()
    for n in nums:
        heap.push(n)
        if len(heap) > k:
            heap.pop()
    return heap.peek()

# --- Test ---
mn = MinHeap()
for v in [5, 3, 8, 1, 9, 2]:
    mn.push(v)
assert mn.pop() == 1
assert mn.pop() == 2
assert mn.peek() == 3

mx = MaxHeap()
for v in [5, 3, 8, 1, 9, 2]:
    mx.push(v)
assert mx.pop() == 9
assert mx.peek() == 8

assert kth_smallest([7, 10, 4, 3, 20, 15], 3) == 7
print("All tests passed.")
''',
    },

    # ── Trie ─────────────────────────────────────────────────────────────────

    {
        "keys": ["trie", "prefix tree", "implement trie", "trie data structure", "autocomplete trie"],
        "lang": "python",
        "title": "Trie (Prefix Tree)",
        "complexity": "Insert/Search/StartsWith: O(m) where m = word length",
        "code": '''\
from __future__ import annotations

class TrieNode:
    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_end = False

class Trie:
    """Prefix tree for O(m) insert, search, and prefix queries.

    Used in autocomplete, spell checkers, and IP routing tables.
    """

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self._traverse(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        return self._traverse(prefix) is not None

    def _traverse(self, s: str) -> TrieNode | None:
        node = self.root
        for ch in s:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def words_with_prefix(self, prefix: str) -> list[str]:
        """Return all words that start with prefix."""
        node = self._traverse(prefix)
        if not node:
            return []
        results: list[str] = []
        self._dfs(node, prefix, results)
        return results

    def _dfs(self, node: TrieNode, path: str, results: list[str]) -> None:
        if node.is_end:
            results.append(path)
        for ch, child in node.children.items():
            self._dfs(child, path + ch, results)

# --- Test ---
t = Trie()
for w in ["apple", "app", "application", "apply", "bat", "ball"]:
    t.insert(w)

assert t.search("apple") == True
assert t.search("app") == True
assert t.search("ap") == False
assert t.starts_with("app") == True
assert t.starts_with("ba") == True
assert t.starts_with("xyz") == False

suggestions = sorted(t.words_with_prefix("app"))
assert suggestions == ["app", "apple", "application", "apply"]
print("All tests passed.")
''',
    },

    # ── Hash Table ───────────────────────────────────────────────────────────

    {
        "keys": ["hash table", "hash map implementation", "implement hash table",
                 "custom hash map", "chaining hash"],
        "lang": "python",
        "title": "Hash Table (Custom — Separate Chaining)",
        "complexity": "Avg O(1) get/put/delete | Worst O(n) with collisions",
        "code": '''\
from typing import TypeVar, Generic, Optional

K = TypeVar("K")
V = TypeVar("V")

class HashTable(Generic[K, V]):
    """Hash table with separate chaining for collision resolution.

    Load factor threshold = 0.75; doubles capacity when exceeded.
    """

    def __init__(self, initial_capacity: int = 16) -> None:
        self._cap = initial_capacity
        self._size = 0
        self._buckets: list[list[tuple]] = [[] for _ in range(self._cap)]

    def _bucket(self, key: K) -> int:
        return hash(key) % self._cap

    def put(self, key: K, value: V) -> None:
        idx = self._bucket(key)
        for i, (k, _) in enumerate(self._buckets[idx]):
            if k == key:
                self._buckets[idx][i] = (key, value)
                return
        self._buckets[idx].append((key, value))
        self._size += 1
        if self._size / self._cap > 0.75:
            self._resize()

    def get(self, key: K) -> Optional[V]:
        for k, v in self._buckets[self._bucket(key)]:
            if k == key:
                return v
        return None

    def delete(self, key: K) -> bool:
        idx = self._bucket(key)
        for i, (k, _) in enumerate(self._buckets[idx]):
            if k == key:
                self._buckets[idx].pop(i)
                self._size -= 1
                return True
        return False

    def _resize(self) -> None:
        old = self._buckets
        self._cap *= 2
        self._buckets = [[] for _ in range(self._cap)]
        self._size = 0
        for bucket in old:
            for k, v in bucket:
                self.put(k, v)

    def __len__(self) -> int:
        return self._size

# --- Test ---
ht: HashTable[str, int] = HashTable()
ht.put("a", 1); ht.put("b", 2); ht.put("c", 3)
assert ht.get("a") == 1
assert ht.get("b") == 2
ht.put("a", 99)
assert ht.get("a") == 99     # update existing key
assert ht.delete("b") == True
assert ht.get("b") is None
assert len(ht) == 2
print("All tests passed.")
''',
    },

    # ── Circular Buffer ──────────────────────────────────────────────────────

    {
        "keys": ["circular buffer", "ring buffer", "circular queue", "fixed size buffer"],
        "lang": "python",
        "title": "Circular Buffer (Ring Buffer)",
        "complexity": "Read/Write: O(1) | Space: O(capacity)",
        "code": '''\
from typing import TypeVar, Generic, Optional

T = TypeVar("T")

class CircularBuffer(Generic[T]):
    """Fixed-size FIFO buffer that overwrites oldest data when full.

    Used in audio processing, network packet buffers, and log rotation.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._buf: list[Optional[T]] = [None] * capacity
        self._head = 0   # read pointer
        self._tail = 0   # write pointer
        self._full = False

    def write(self, item: T) -> None:
        self._buf[self._tail] = item
        if self._full:
            self._head = (self._head + 1) % self.capacity   # overwrite oldest
        self._tail = (self._tail + 1) % self.capacity
        self._full = self._head == self._tail

    def read(self) -> Optional[T]:
        if self.is_empty():
            return None
        val = self._buf[self._head]
        self._full = False
        self._head = (self._head + 1) % self.capacity
        return val

    def is_empty(self) -> bool:
        return not self._full and self._head == self._tail

    def is_full(self) -> bool:
        return self._full

    def __len__(self) -> int:
        if self._full:
            return self.capacity
        return (self._tail - self._head) % self.capacity

# --- Test ---
cb: CircularBuffer[int] = CircularBuffer(3)
cb.write(1); cb.write(2); cb.write(3)
assert cb.is_full()
cb.write(4)                 # overwrites 1
assert cb.read() == 2       # oldest remaining
assert cb.read() == 3
assert cb.read() == 4
assert cb.is_empty()
print("All tests passed.")
''',
    },

    # ── Sorting: Bubble Sort ─────────────────────────────────────────────────

    {
        "keys": ["bubble sort", "implement bubble sort", "bubblesort"],
        "lang": "python",
        "title": "Bubble Sort",
        "complexity": "Time: O(n²) avg/worst, O(n) best (sorted) | Space: O(1)",
        "code": '''\
def bubble_sort(arr: list[int]) -> list[int]:
    """Repeatedly swap adjacent elements that are out of order.

    Stable sort. Optimised with early-exit when no swaps occur in a pass.
    Best case O(n) on already-sorted input.
    """
    n = len(arr)
    arr = arr[:]
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break   # already sorted
    return arr

# --- Test ---
assert bubble_sort([64, 34, 25, 12, 22, 11, 90]) == [11, 12, 22, 25, 34, 64, 90]
assert bubble_sort([]) == []
assert bubble_sort([1]) == [1]
assert bubble_sort([1, 2, 3]) == [1, 2, 3]   # already sorted — early exit
print("All tests passed.")
''',
    },

    # ── Sorting: Insertion Sort ───────────────────────────────────────────────

    {
        "keys": ["insertion sort", "implement insertion sort"],
        "lang": "python",
        "title": "Insertion Sort",
        "complexity": "Time: O(n²) avg/worst, O(n) best | Space: O(1) | Stable",
        "code": '''\
def insertion_sort(arr: list[int]) -> list[int]:
    """Build sorted portion one element at a time by shifting larger elements right.

    Excellent for small arrays and nearly-sorted data. Online algorithm.
    """
    arr = arr[:]
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# --- Test ---
assert insertion_sort([12, 11, 13, 5, 6]) == [5, 6, 11, 12, 13]
assert insertion_sort([]) == []
assert insertion_sort([1]) == [1]
assert insertion_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]
print("All tests passed.")
''',
    },

    # ── Sorting: Selection Sort ───────────────────────────────────────────────

    {
        "keys": ["selection sort", "implement selection sort"],
        "lang": "python",
        "title": "Selection Sort",
        "complexity": "Time: O(n²) all cases | Space: O(1) | Not stable",
        "code": '''\
def selection_sort(arr: list[int]) -> list[int]:
    """Find the minimum in the unsorted portion and place it at the front.

    Makes exactly n-1 swaps — good when writes are expensive (e.g. flash memory).
    """
    arr = arr[:]
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# --- Test ---
assert selection_sort([64, 25, 12, 22, 11]) == [11, 12, 22, 25, 64]
assert selection_sort([]) == []
assert selection_sort([3, 2, 1]) == [1, 2, 3]
print("All tests passed.")
''',
    },

    # ── Sorting: Heap Sort ────────────────────────────────────────────────────

    {
        "keys": ["heap sort", "heapsort", "implement heap sort"],
        "lang": "python",
        "title": "Heap Sort",
        "complexity": "Time: O(n log n) all cases | Space: O(1) | Not stable",
        "code": '''\
def heap_sort(arr: list[int]) -> list[int]:
    """In-place sort using a max-heap. Guaranteed O(n log n) with O(1) extra space."""
    arr = arr[:]
    n = len(arr)

    def heapify(n: int, i: int) -> None:
        largest = i
        left, right = 2 * i + 1, 2 * i + 2
        if left  < n and arr[left]  > arr[largest]: largest = left
        if right < n and arr[right] > arr[largest]: largest = right
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(n, largest)

    for i in range(n // 2 - 1, -1, -1):   # build max-heap
        heapify(n, i)
    for i in range(n - 1, 0, -1):          # extract elements
        arr[0], arr[i] = arr[i], arr[0]
        heapify(i, 0)
    return arr

# --- Test ---
assert heap_sort([12, 11, 13, 5, 6, 7]) == [5, 6, 7, 11, 12, 13]
assert heap_sort([]) == []
assert heap_sort([1]) == [1]
print("All tests passed.")
''',
    },

    # ── Sorting: Counting Sort ────────────────────────────────────────────────

    {
        "keys": ["counting sort", "implement counting sort"],
        "lang": "python",
        "title": "Counting Sort",
        "complexity": "Time: O(n + k) | Space: O(k) where k = value range",
        "code": '''\
def counting_sort(arr: list[int]) -> list[int]:
    """Non-comparison sort. Optimal for small integer ranges.

    Stable: equal elements maintain relative order.
    Only works on non-negative integers (or integers with a known min).
    """
    if not arr:
        return []
    min_val, max_val = min(arr), max(arr)
    k = max_val - min_val + 1
    counts = [0] * k
    for x in arr:
        counts[x - min_val] += 1
    result: list[int] = []
    for i, c in enumerate(counts):
        result.extend([i + min_val] * c)
    return result

# --- Test ---
assert counting_sort([4, 2, 2, 8, 3, 3, 1]) == [1, 2, 2, 3, 3, 4, 8]
assert counting_sort([]) == []
assert counting_sort([1]) == [1]
assert counting_sort([3, 1, 2]) == [1, 2, 3]
print("All tests passed.")
''',
    },

    # ── Sorting: Radix Sort ───────────────────────────────────────────────────

    {
        "keys": ["radix sort", "implement radix sort"],
        "lang": "python",
        "title": "Radix Sort (LSD)",
        "complexity": "Time: O(d * (n + k)) | Space: O(n + k) where d = digits, k = base",
        "code": '''\
def radix_sort(arr: list[int]) -> list[int]:
    """Least-Significant-Digit radix sort for non-negative integers.

    Processes one digit position at a time using stable counting sort.
    """
    if not arr:
        return []
    arr = arr[:]
    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        arr = _counting_by_digit(arr, exp)
        exp *= 10
    return arr

def _counting_by_digit(arr: list[int], exp: int) -> list[int]:
    n = len(arr)
    output = [0] * n
    count  = [0] * 10
    for x in arr:
        count[(x // exp) % 10] += 1
    for i in range(1, 10):
        count[i] += count[i - 1]
    for i in range(n - 1, -1, -1):
        digit = (arr[i] // exp) % 10
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1
    return output

# --- Test ---
assert radix_sort([170, 45, 75, 90, 802, 24, 2, 66]) == [2, 24, 45, 66, 75, 90, 170, 802]
assert radix_sort([]) == []
assert radix_sort([3, 1, 2]) == [1, 2, 3]
print("All tests passed.")
''',
    },

    # ── Search: Linear Search ────────────────────────────────────────────────

    {
        "keys": ["linear search", "sequential search", "implement linear search"],
        "lang": "python",
        "title": "Linear Search",
        "complexity": "Time: O(n) | Space: O(1)",
        "code": '''\
from typing import TypeVar, Optional

T = TypeVar("T")

def linear_search(arr: list[T], target: T) -> int:
    """Scan every element. Works on unsorted data. Returns index or -1."""
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

def linear_search_all(arr: list[T], target: T) -> list[int]:
    """Return all indices where target appears."""
    return [i for i, val in enumerate(arr) if val == target]

# --- Test ---
assert linear_search([3, 1, 4, 1, 5, 9], 5) == 4
assert linear_search([3, 1, 4, 1, 5, 9], 7) == -1
assert linear_search([], 1) == -1
assert linear_search_all([1, 2, 1, 3, 1], 1) == [0, 2, 4]
print("All tests passed.")
''',
    },

    # ── Search: Jump Search ───────────────────────────────────────────────────

    {
        "keys": ["jump search", "block search", "implement jump search"],
        "lang": "python",
        "title": "Jump Search",
        "complexity": "Time: O(√n) | Space: O(1) | Requires sorted array",
        "code": '''\
import math

def jump_search(arr: list[int], target: int) -> int:
    """Jump ahead by √n steps, then linear scan backward.

    Better than linear (O(n)) but slower than binary (O(log n)).
    Good for sorted arrays where backward traversal is expensive.
    """
    n = len(arr)
    if n == 0:
        return -1
    step = int(math.sqrt(n))
    prev = 0
    while prev < n and arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    for i in range(prev, min(step, n)):
        if arr[i] == target:
            return i
    return -1

# --- Test ---
arr = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
assert jump_search(arr, 55) == 10
assert jump_search(arr, 3)  == 4
assert jump_search(arr, 7)  == -1
assert jump_search([], 1)   == -1
print("All tests passed.")
''',
    },

    # ── Search: Interpolation Search ─────────────────────────────────────────

    {
        "keys": ["interpolation search", "implement interpolation search"],
        "lang": "python",
        "title": "Interpolation Search",
        "complexity": "Time: O(log log n) avg for uniform data, O(n) worst | Space: O(1)",
        "code": '''\
def interpolation_search(arr: list[int], target: int) -> int:
    """Estimate position by interpolating between low and high values.

    Outperforms binary search on uniformly distributed sorted arrays.
    Degrades to O(n) on skewed distributions.
    """
    lo, hi = 0, len(arr) - 1
    while lo <= hi and arr[lo] <= target <= arr[hi]:
        if arr[lo] == arr[hi]:
            return lo if arr[lo] == target else -1
        pos = lo + ((target - arr[lo]) * (hi - lo)) // (arr[hi] - arr[lo])
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            lo = pos + 1
        else:
            hi = pos - 1
    return -1

# --- Test ---
arr = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
assert interpolation_search(arr, 70)  == 6
assert interpolation_search(arr, 10)  == 0
assert interpolation_search(arr, 100) == 9
assert interpolation_search(arr, 55)  == -1
print("All tests passed.")
''',
    },

    # ── Graph: Dijkstra ──────────────────────────────────────────────────────

    {
        "keys": ["dijkstra", "dijkstra algorithm", "shortest path weighted", "weighted shortest path"],
        "lang": "python",
        "title": "Dijkstra's Shortest Path",
        "complexity": "Time: O((V + E) log V) | Space: O(V)",
        "code": '''\
import heapq
from typing import Optional

Graph = dict[str, list[tuple[str, float]]]  # node -> [(neighbour, weight)]

def dijkstra(graph: Graph, start: str) -> dict[str, float]:
    """Return shortest distances from start to all reachable nodes.

    Uses a min-heap (priority queue) for greedy selection.
    Handles directed/undirected weighted graphs with non-negative edges.
    """
    dist: dict[str, float] = {start: 0.0}
    heap: list[tuple[float, str]] = [(0.0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue   # stale entry
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist

def shortest_path_dijkstra(graph: Graph, start: str, end: str) -> Optional[list[str]]:
    """Return the actual path (node list) from start to end, or None."""
    dist: dict[str, float] = {start: 0.0}
    prev: dict[str, Optional[str]] = {start: None}
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        if u == end:
            path: list[str] = []
            cur: Optional[str] = end
            while cur is not None:
                path.append(cur)
                cur = prev.get(cur)
            return path[::-1]
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return None

# --- Test ---
g: Graph = {
    "A": [("B", 4), ("C", 2)],
    "B": [("D", 3), ("C", 1)],
    "C": [("B", 1), ("D", 5)],
    "D": [],
}
distances = dijkstra(g, "A")
assert distances["A"] == 0
assert distances["C"] == 2
assert distances["B"] == 3   # A->C->B = 2+1 = 3 (cheaper than A->B=4)
assert distances["D"] == 6   # A->C->B->D = 2+1+3 = 6

path = shortest_path_dijkstra(g, "A", "D")
assert path == ["A", "C", "B", "D"]
print("All tests passed.")
''',
    },

    # ── Graph: Topological Sort ───────────────────────────────────────────────

    {
        "keys": ["topological sort", "topological ordering", "dag sort", "dependency order"],
        "lang": "python",
        "title": "Topological Sort (Kahn's Algorithm + DFS)",
        "complexity": "Time: O(V + E) | Space: O(V)",
        "code": '''\
from collections import deque

def topological_sort_kahn(graph: dict[str, list[str]]) -> list[str]:
    """Kahn\'s algorithm using BFS and in-degree counts.

    Returns a valid topological ordering, or [] if a cycle is detected.
    """
    in_degree: dict[str, int] = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] = in_degree.get(v, 0) + 1
            if v not in in_degree:
                in_degree[v] = in_degree.get(v, 0)

    queue = deque(u for u, d in in_degree.items() if d == 0)
    order: list[str] = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return order if len(order) == len(in_degree) else []  # [] = cycle

def topological_sort_dfs(graph: dict[str, list[str]]) -> list[str]:
    """DFS-based topological sort using finish-time stack."""
    visited: set[str] = set()
    stack: list[str] = []

    def dfs(node: str) -> None:
        visited.add(node)
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                dfs(neighbour)
        stack.append(node)

    for node in graph:
        if node not in visited:
            dfs(node)
    return stack[::-1]

# --- Test ---
dag = {
    "A": ["C"],
    "B": ["C", "D"],
    "C": ["E"],
    "D": ["F"],
    "E": ["F"],
    "F": [],
}
order = topological_sort_kahn(dag)
# Verify A before C, C before E, E before F, B before D, D before F
assert order.index("A") < order.index("C")
assert order.index("C") < order.index("E")
assert order.index("E") < order.index("F")

order_dfs = topological_sort_dfs(dag)
assert order_dfs.index("A") < order_dfs.index("F")
print("All tests passed.")
''',
    },

    # ── Graph: Union-Find ─────────────────────────────────────────────────────

    {
        "keys": ["union find", "disjoint set", "union find data structure", "path compression"],
        "lang": "python",
        "title": "Union-Find (Disjoint Set Union)",
        "complexity": "Union/Find: O(α(n)) amortised (inverse Ackermann — effectively O(1))",
        "code": '''\
class UnionFind:
    """Disjoint Set Union with path compression and union by rank.

    Used for Kruskal\'s MST, cycle detection, and network connectivity.
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank   = [0] * n
        self.components = n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False   # already connected
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.components -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

# --- Test ---
uf = UnionFind(6)
uf.union(0, 1); uf.union(1, 2)
uf.union(3, 4)
assert uf.connected(0, 2) == True
assert uf.connected(0, 3) == False
assert uf.components == 3   # {0,1,2}, {3,4}, {5}
uf.union(2, 3)
assert uf.connected(0, 4) == True
assert uf.components == 2
print("All tests passed.")
''',
    },

    # ── Graph: Cycle Detection ────────────────────────────────────────────────

    {
        "keys": ["cycle detection", "detect cycle", "cycle in graph", "has cycle"],
        "lang": "python",
        "title": "Cycle Detection (Directed and Undirected Graphs)",
        "complexity": "Time: O(V + E) | Space: O(V)",
        "code": '''\
def has_cycle_directed(graph: dict[str, list[str]]) -> bool:
    """DFS with colour marking: white=unvisited, grey=in-stack, black=done.

    A grey->grey back-edge means a cycle.
    """
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {u: WHITE for u in graph}

    def dfs(node: str) -> bool:
        colour[node] = GREY
        for neighbour in graph.get(node, []):
            if colour.get(neighbour, WHITE) == GREY:
                return True   # back edge -> cycle
            if colour.get(neighbour, WHITE) == WHITE:
                if dfs(neighbour):
                    return True
        colour[node] = BLACK
        return False

    return any(dfs(u) for u in graph if colour[u] == WHITE)

def has_cycle_undirected(graph: dict[str, list[str]]) -> bool:
    """Union-Find approach for undirected graphs."""
    nodes = list(graph.keys())
    idx   = {n: i for i, n in enumerate(nodes)}
    uf    = list(range(len(nodes)))

    def find(x: int) -> int:
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    for u in graph:
        for v in graph[u]:
            if u < v:   # undirected: process each edge once
                ru, rv = find(idx[u]), find(idx[v])
                if ru == rv:
                    return True
                uf[ru] = rv
    return False

# --- Test ---
dag_no_cycle   = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
dag_with_cycle = {"A": ["B"], "B": ["C"], "C": ["A"]}
assert has_cycle_directed(dag_no_cycle)   == False
assert has_cycle_directed(dag_with_cycle) == True

ug_no_cycle   = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
ug_with_cycle = {"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}
assert has_cycle_undirected(ug_no_cycle)   == False
assert has_cycle_undirected(ug_with_cycle) == True
print("All tests passed.")
''',
    },

    # ── DP: Knapsack 0/1 ──────────────────────────────────────────────────────

    {
        "keys": ["knapsack", "0/1 knapsack", "01 knapsack", "knapsack problem"],
        "lang": "python",
        "title": "0/1 Knapsack (Dynamic Programming)",
        "complexity": "Time: O(n * W) | Space: O(W) optimised",
        "code": '''\
def knapsack(weights: list[int], values: list[int], capacity: int) -> int:
    """Return the maximum value achievable within weight capacity.

    Each item can be taken at most once (0/1).
    Uses 1D DP table iterated in reverse to avoid using an item twice.
    """
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]

def knapsack_with_items(
    weights: list[int], values: list[int], capacity: int
) -> tuple[int, list[int]]:
    """Also return which items were selected."""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
    # Backtrack
    selected: list[int] = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= weights[i - 1]
    return dp[n][capacity], selected[::-1]

# --- Test ---
weights = [1, 3, 4, 5]
values  = [1, 4, 5, 7]
assert knapsack(weights, values, 7) == 9

val, items = knapsack_with_items(weights, values, 7)
assert val == 9
assert set(items) == {1, 2}   # items at index 1 (w=3,v=4) and 2 (w=4,v=5)
print("All tests passed.")
''',
    },

    # ── DP: Longest Common Subsequence ────────────────────────────────────────

    {
        "keys": ["longest common subsequence", "lcs", "lcs algorithm", "common subsequence"],
        "lang": "python",
        "title": "Longest Common Subsequence (LCS)",
        "complexity": "Time: O(m * n) | Space: O(m * n)",
        "code": '''\
def lcs_length(s1: str, s2: str) -> int:
    """Return the length of the longest common subsequence."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def lcs_string(s1: str, s2: str) -> str:
    """Return the actual LCS string by backtracking the DP table."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    result: list[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            result.append(s1[i - 1])
            i -= 1; j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(result))

# --- Test ---
assert lcs_length("ABCBDAB", "BDCAB") == 4
assert lcs_string("ABCBDAB", "BDCAB") == "BCAB"
assert lcs_length("", "ABC") == 0
assert lcs_length("ABC", "ABC") == 3
print("All tests passed.")
''',
    },

    # ── DP: Longest Increasing Subsequence ────────────────────────────────────

    {
        "keys": ["longest increasing subsequence", "lis", "lis algorithm"],
        "lang": "python",
        "title": "Longest Increasing Subsequence (LIS)",
        "complexity": "DP: O(n²) | Binary search: O(n log n)",
        "code": '''\
import bisect

def lis_length_dp(nums: list[int]) -> int:
    """O(n²) DP approach — easier to understand."""
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def lis_length_fast(nums: list[int]) -> int:
    """O(n log n) patience sorting approach."""
    tails: list[int] = []
    for x in nums:
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)

# --- Test ---
assert lis_length_dp([10, 9, 2, 5, 3, 7, 101, 18]) == 4
assert lis_length_fast([10, 9, 2, 5, 3, 7, 101, 18]) == 4
assert lis_length_fast([0, 1, 0, 3, 2, 3]) == 4
assert lis_length_fast([7, 7, 7]) == 1
print("All tests passed.")
''',
    },

    # ── DP: Coin Change ───────────────────────────────────────────────────────

    {
        "keys": ["coin change", "minimum coins", "coin change problem"],
        "lang": "python",
        "title": "Coin Change (Minimum Coins)",
        "complexity": "Time: O(amount * len(coins)) | Space: O(amount)",
        "code": '''\
def coin_change(coins: list[int], amount: int) -> int:
    """Return minimum number of coins to make amount, or -1 if impossible."""
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
        for coin in coins:
            if coin <= a:
                dp[a] = min(dp[a], dp[a - coin] + 1)
    return int(dp[amount]) if dp[amount] != float("inf") else -1

def coin_change_ways(coins: list[int], amount: int) -> int:
    """Return the number of distinct combinations that make up amount."""
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for a in range(coin, amount + 1):
            dp[a] += dp[a - coin]
    return dp[amount]

# --- Test ---
assert coin_change([1, 5, 6, 9], 11) == 2     # 5+6
assert coin_change([2], 3)           == -1    # impossible
assert coin_change([1, 2, 5], 11)    == 3     # 5+5+1
assert coin_change_ways([1, 2, 5], 5) == 4    # {5},{2+2+1},{2+1+1+1},{1*5}
print("All tests passed.")
''',
    },

    # ── DP: Edit Distance ─────────────────────────────────────────────────────

    {
        "keys": ["edit distance", "levenshtein", "levenshtein distance", "string edit distance"],
        "lang": "python",
        "title": "Edit Distance (Levenshtein)",
        "complexity": "Time: O(m * n) | Space: O(min(m, n))",
        "code": '''\
def edit_distance(s1: str, s2: str) -> int:
    """Minimum insert/delete/replace operations to transform s1 into s2."""
    m, n = len(s1), len(s2)
    if m < n:
        s1, s2, m, n = s2, s1, n, m   # ensure s1 is longer for space opt
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j],       # delete
                                   curr[j - 1],   # insert
                                   prev[j - 1])   # replace
        prev = curr
    return prev[n]

# --- Test ---
assert edit_distance("kitten", "sitting") == 3
assert edit_distance("", "abc")           == 3
assert edit_distance("abc", "abc")        == 0
assert edit_distance("horse", "ros")      == 3
print("All tests passed.")
''',
    },

    # ── DP: Kadane's Max Subarray ─────────────────────────────────────────────

    {
        "keys": ["kadane", "max subarray", "maximum subarray", "largest sum subarray",
                 "kadane algorithm"],
        "lang": "python",
        "title": "Kadane's Maximum Subarray",
        "complexity": "Time: O(n) | Space: O(1)",
        "code": '''\
def max_subarray(nums: list[int]) -> int:
    """Return the maximum sum of any contiguous subarray (Kadane\'s algorithm)."""
    if not nums:
        return 0
    max_sum = cur_sum = nums[0]
    for x in nums[1:]:
        cur_sum = max(x, cur_sum + x)
        max_sum = max(max_sum, cur_sum)
    return max_sum

def max_subarray_with_indices(nums: list[int]) -> tuple[int, int, int]:
    """Return (max_sum, start_index, end_index)."""
    max_sum = cur_sum = nums[0]
    start = end = 0
    temp_start = 0
    for i in range(1, len(nums)):
        if cur_sum + nums[i] < nums[i]:
            cur_sum = nums[i]
            temp_start = i
        else:
            cur_sum += nums[i]
        if cur_sum > max_sum:
            max_sum = cur_sum
            start = temp_start
            end = i
    return max_sum, start, end

# --- Test ---
assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6   # [4,-1,2,1]
assert max_subarray([-1, -2, -3]) == -1
assert max_subarray([1]) == 1
s, lo, hi = max_subarray_with_indices([-2, 1, -3, 4, -1, 2, 1, -5, 4])
assert s == 6 and lo == 3 and hi == 6
print("All tests passed.")
''',
    },

    # ── Strings: Anagram Check ───────────────────────────────────────────────

    {
        "keys": ["anagram", "check anagram", "is anagram", "anagram check"],
        "lang": "python",
        "title": "Anagram Check",
        "complexity": "Time: O(n) | Space: O(k) where k = character set size",
        "code": '''\
from collections import Counter

def is_anagram(s1: str, s2: str, ignore_spaces: bool = True) -> bool:
    """Return True if s1 and s2 are anagrams (same letters, different order)."""
    if ignore_spaces:
        s1 = s1.replace(" ", "").lower()
        s2 = s2.replace(" ", "").lower()
    return Counter(s1) == Counter(s2)

def group_anagrams(words: list[str]) -> list[list[str]]:
    """Group a list of words into anagram clusters."""
    groups: dict[tuple, list[str]] = {}
    for word in words:
        key = tuple(sorted(word.lower()))
        groups.setdefault(key, []).append(word)
    return list(groups.values())

# --- Test ---
assert is_anagram("listen", "silent") == True
assert is_anagram("hello", "world")   == False
assert is_anagram("Astronomer", "Moon starer") == True

groups = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
anagram_sets = [sorted(g) for g in groups]
assert sorted(["ate", "eat", "tea"]) in anagram_sets
assert sorted(["nat", "tan"]) in anagram_sets
print("All tests passed.")
''',
    },

    # ── Strings: Run-Length Encoding ──────────────────────────────────────────

    {
        "keys": ["run length encoding", "rle", "rle compression", "run length compress"],
        "lang": "python",
        "title": "Run-Length Encoding (RLE)",
        "complexity": "Encode/Decode: O(n)",
        "code": '''\
def rle_encode(s: str) -> str:
    """Compress consecutive repeated characters: "aaabbc" -> "3a2bc"."""
    if not s:
        return ""
    result: list[str] = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            result.append(("" if count == 1 else str(count)) + s[i - 1])
            count = 1
    result.append(("" if count == 1 else str(count)) + s[-1])
    return "".join(result)

def rle_decode(s: str) -> str:
    """Decompress RLE string: "3a2bc" -> "aaabbc"."""
    result: list[str] = []
    i = 0
    while i < len(s):
        count_str = ""
        while i < len(s) and s[i].isdigit():
            count_str += s[i]
            i += 1
        if i < len(s):
            count = int(count_str) if count_str else 1
            result.append(s[i] * count)
            i += 1
    return "".join(result)

# --- Test ---
assert rle_encode("aaabbc")   == "3a2bc"
assert rle_encode("aabbcc")   == "2a2b2c"
assert rle_encode("abc")      == "abc"
assert rle_decode("3a2bc")    == "aaabbc"
assert rle_decode(rle_encode("aaabbbcccc")) == "aaabbbcccc"
print("All tests passed.")
''',
    },

    # ── Strings: KMP Pattern Match ────────────────────────────────────────────

    {
        "keys": ["kmp", "knuth morris pratt", "kmp pattern", "string pattern match",
                 "pattern matching kmp"],
        "lang": "python",
        "title": "KMP Pattern Matching",
        "complexity": "Preprocessing: O(m) | Search: O(n) | Total: O(n + m)",
        "code": '''\
def kmp_search(text: str, pattern: str) -> list[int]:
    """Return all starting indices where pattern appears in text.

    KMP avoids redundant comparisons by precomputing a failure function.
    """
    n, m = len(text), len(pattern)
    if m == 0:
        return []
    lps = _build_lps(pattern)
    matches: list[int] = []
    i = j = 0
    while i < n:
        if text[i] == pattern[j]:
            i += 1; j += 1
            if j == m:
                matches.append(i - m)
                j = lps[j - 1]
        else:
            if j:
                j = lps[j - 1]
            else:
                i += 1
    return matches

def _build_lps(pattern: str) -> list[int]:
    """Longest proper prefix which is also suffix — the failure function."""
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1
    return lps

# --- Test ---
assert kmp_search("aabaacaadaabaaba", "aaba") == [0, 9, 12]
assert kmp_search("hello world", "world") == [6]
assert kmp_search("aaa", "aa") == [0, 1]
assert kmp_search("abc", "xyz") == []
print("All tests passed.")
''',
    },

    # ── Strings: Permutations ─────────────────────────────────────────────────

    {
        "keys": ["string permutation", "permutations of string", "all permutations",
                 "generate permutations"],
        "lang": "python",
        "title": "String Permutations",
        "complexity": "Time: O(n! * n) | Space: O(n!)",
        "code": '''\
def permutations(s: str) -> list[str]:
    """Return all unique permutations of string s."""
    if len(s) <= 1:
        return [s]
    result: set[str] = set()
    for i, ch in enumerate(s):
        rest = s[:i] + s[i+1:]
        for perm in permutations(rest):
            result.add(ch + perm)
    return sorted(result)

def permutations_iterative(s: str) -> list[str]:
    """Heap\'s algorithm — iterative, generates n! permutations in-place."""
    arr = list(s)
    n = len(arr)
    result = []
    c = [0] * n
    result.append("".join(arr))
    i = 0
    while i < n:
        if c[i] < i:
            if i % 2 == 0:
                arr[0], arr[i] = arr[i], arr[0]
            else:
                arr[c[i]], arr[i] = arr[i], arr[c[i]]
            result.append("".join(arr))
            c[i] += 1
            i = 0
        else:
            c[i] = 0
            i += 1
    return sorted(set(result))

# --- Test ---
assert permutations("abc") == ["abc", "acb", "bac", "bca", "cab", "cba"]
assert permutations("aa")  == ["aa"]
assert sorted(permutations_iterative("abc")) == sorted(permutations("abc"))
print("All tests passed.")
''',
    },

    # ── Strings: Word Count ───────────────────────────────────────────────────

    {
        "keys": ["word count", "word frequency", "count words", "word counter"],
        "lang": "python",
        "title": "Word Frequency Counter",
        "complexity": "Time: O(n) | Space: O(k) where k = unique words",
        "code": '''\
import re
from collections import Counter

def word_frequency(text: str, top_n: int = 10) -> list[tuple[str, int]]:
    """Count word occurrences, return top_n most common words."""
    words = re.findall(r"[a-zA-Z\']+", text.lower())
    return Counter(words).most_common(top_n)

def word_count(text: str) -> int:
    """Return total number of words in text."""
    return len(re.findall(r"\\S+", text.strip()))

# --- Test ---
text = "the quick brown fox jumps over the lazy dog the fox"
freq = dict(word_frequency(text))
assert freq["the"] == 3
assert freq["fox"] == 2
assert word_count("hello world foo") == 3
assert word_count("") == 0
print("All tests passed.")
''',
    },

    # ── Math: GCD / LCM ──────────────────────────────────────────────────────

    {
        "keys": ["gcd", "lcm", "greatest common divisor", "least common multiple",
                 "euclidean algorithm"],
        "lang": "python",
        "title": "GCD and LCM (Euclidean Algorithm)",
        "complexity": "GCD: O(log min(a,b)) | LCM: O(log min(a,b))",
        "code": '''\
from math import gcd as _gcd
from functools import reduce
from typing import Sequence

def gcd(a: int, b: int) -> int:
    """Greatest common divisor via Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return abs(a)

def lcm(a: int, b: int) -> int:
    """Least common multiple: |a * b| / gcd(a, b)."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)

def gcd_multiple(nums: Sequence[int]) -> int:
    """GCD of a list of integers."""
    return reduce(gcd, nums)

def lcm_multiple(nums: Sequence[int]) -> int:
    """LCM of a list of integers."""
    return reduce(lcm, nums)

# --- Test ---
assert gcd(48, 18) == 6
assert gcd(100, 75) == 25
assert lcm(4, 6)   == 12
assert lcm(3, 5)   == 15
assert gcd_multiple([12, 18, 24]) == 6
assert lcm_multiple([4, 6, 10])   == 60
print("All tests passed.")
''',
    },

    # ── Math: Fast Power ─────────────────────────────────────────────────────

    {
        "keys": ["fast power", "exponentiation by squaring", "power function", "fast exponent"],
        "lang": "python",
        "title": "Fast Power (Exponentiation by Squaring)",
        "complexity": "Time: O(log n) | Space: O(1)",
        "code": '''\
def fast_power(base: int, exp: int, mod: int = None) -> int:
    """Compute base^exp in O(log exp) using repeated squaring.

    Optional mod parameter for modular exponentiation (cryptography use).
    """
    if exp < 0:
        raise ValueError("Negative exponents not supported for integers")
    result = 1
    while exp > 0:
        if exp % 2 == 1:
            result = result * base if mod is None else result * base % mod
        base = base * base if mod is None else base * base % mod
        exp //= 2
    return result

# --- Test ---
assert fast_power(2, 10)       == 1024
assert fast_power(3, 0)        == 1
assert fast_power(2, 32)       == 4294967296
assert fast_power(2, 10, 1000) == 24    # 1024 % 1000
assert fast_power(5, 3)        == 125
print("All tests passed.")
''',
    },

    # ── Math: Prime Factorization ─────────────────────────────────────────────

    {
        "keys": ["prime factorization", "prime factors", "factorize number"],
        "lang": "python",
        "title": "Prime Factorization",
        "complexity": "Time: O(√n) | Space: O(log n)",
        "code": '''\
def prime_factors(n: int) -> list[int]:
    """Return sorted list of prime factors of n (with repetition)."""
    if n < 2:
        return []
    factors: list[int] = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def prime_factor_dict(n: int) -> dict[int, int]:
    """Return {prime: exponent} factorization."""
    result: dict[int, int] = {}
    for p in prime_factors(n):
        result[p] = result.get(p, 0) + 1
    return result

# --- Test ---
assert prime_factors(12)       == [2, 2, 3]
assert prime_factors(100)      == [2, 2, 5, 5]
assert prime_factors(13)       == [13]
assert prime_factor_dict(360)  == {2: 3, 3: 2, 5: 1}
print("All tests passed.")
''',
    },

    # ── Math: Number Conversion ───────────────────────────────────────────────

    {
        "keys": ["number to binary", "number to hex", "base conversion", "decimal to binary",
                 "convert to binary", "convert to hex"],
        "lang": "python",
        "title": "Number Base Conversion (Binary, Hex, Octal)",
        "complexity": "Time: O(log n) | Space: O(log n)",
        "code": '''\
def to_binary(n: int) -> str:
    """Convert non-negative integer to binary string (no 0b prefix)."""
    if n == 0: return "0"
    bits: list[str] = []
    x = abs(n)
    while x:
        bits.append(str(x & 1))
        x >>= 1
    return ("-" if n < 0 else "") + "".join(reversed(bits))

def to_hex(n: int) -> str:
    """Convert integer to lowercase hex string (no 0x prefix)."""
    if n == 0: return "0"
    digits = "0123456789abcdef"
    result: list[str] = []
    x = abs(n)
    while x:
        result.append(digits[x & 0xF])
        x >>= 4
    return ("-" if n < 0 else "") + "".join(reversed(result))

def from_base(s: str, base: int) -> int:
    """Convert string representation in given base to integer."""
    return int(s, base)

# --- Test ---
assert to_binary(10)   == "1010"
assert to_binary(255)  == "11111111"
assert to_binary(0)    == "0"
assert to_hex(255)     == "ff"
assert to_hex(16)      == "10"
assert from_base("1010", 2) == 10
assert from_base("ff", 16)  == 255
print("All tests passed.")
''',
    },

    # ── OOP: Factory Pattern ──────────────────────────────────────────────────

    {
        "keys": ["factory pattern", "factory method", "implement factory", "factory design pattern"],
        "lang": "python",
        "title": "Factory Pattern",
        "complexity": "O(1) object creation",
        "code": '''\
from __future__ import annotations
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...
    @abstractmethod
    def perimeter(self) -> float: ...
    def describe(self) -> str:
        return f"{self.__class__.__name__}: area={self.area():.2f}, perimeter={self.perimeter():.2f}"

class Circle(Shape):
    def __init__(self, radius: float) -> None:
        self.radius = radius
    def area(self) -> float:
        import math; return math.pi * self.radius ** 2
    def perimeter(self) -> float:
        import math; return 2 * math.pi * self.radius

class Rectangle(Shape):
    def __init__(self, width: float, height: float) -> None:
        self.width = width; self.height = height
    def area(self) -> float: return self.width * self.height
    def perimeter(self) -> float: return 2 * (self.width + self.height)

class Triangle(Shape):
    def __init__(self, a: float, b: float, c: float) -> None:
        self.a = a; self.b = b; self.c = c
    def area(self) -> float:
        s = (self.a + self.b + self.c) / 2
        return (s * (s-self.a) * (s-self.b) * (s-self.c)) ** 0.5
    def perimeter(self) -> float: return self.a + self.b + self.c

class ShapeFactory:
    """Create shapes by name — decouples client from concrete classes."""
    _registry: dict[str, type] = {
        "circle":    Circle,
        "rectangle": Rectangle,
        "triangle":  Triangle,
    }

    @classmethod
    def create(cls, shape_type: str, **kwargs) -> Shape:
        klass = cls._registry.get(shape_type.lower())
        if not klass:
            raise ValueError(f"Unknown shape: {shape_type!r}")
        return klass(**kwargs)

    @classmethod
    def register(cls, name: str, klass: type) -> None:
        cls._registry[name] = klass

# --- Test ---
c = ShapeFactory.create("circle", radius=5)
r = ShapeFactory.create("rectangle", width=4, height=6)
import math
assert abs(c.area() - math.pi * 25) < 1e-9
assert r.area() == 24
assert r.perimeter() == 20
try:
    ShapeFactory.create("hexagon")
    assert False
except ValueError:
    pass
print("All tests passed.")
''',
    },

    # ── OOP: Observer Pattern ─────────────────────────────────────────────────

    {
        "keys": ["observer pattern", "event listener pattern", "pub sub pattern",
                 "publish subscribe", "observer design"],
        "lang": "python",
        "title": "Observer Pattern (Event System)",
        "complexity": "Subscribe: O(1) | Notify: O(n listeners)",
        "code": '''\
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

class Observer(ABC):
    @abstractmethod
    def update(self, event: str, data: Any) -> None: ...

class EventEmitter:
    """Subject that maintains a list of observers per event type."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Observer]] = {}

    def subscribe(self, event: str, observer: Observer) -> None:
        self._listeners.setdefault(event, []).append(observer)

    def unsubscribe(self, event: str, observer: Observer) -> None:
        if event in self._listeners:
            self._listeners[event] = [o for o in self._listeners[event] if o is not observer]

    def emit(self, event: str, data: Any = None) -> None:
        for obs in self._listeners.get(event, []):
            obs.update(event, data)

class LogObserver(Observer):
    def __init__(self) -> None:
        self.log: list[tuple[str, Any]] = []
    def update(self, event: str, data: Any) -> None:
        self.log.append((event, data))

# --- Test ---
emitter = EventEmitter()
logger1 = LogObserver()
logger2 = LogObserver()

emitter.subscribe("login",  logger1)
emitter.subscribe("login",  logger2)
emitter.subscribe("logout", logger1)

emitter.emit("login",  {"user": "alice"})
emitter.emit("logout", {"user": "alice"})
emitter.emit("login",  {"user": "bob"})

assert len(logger1.log) == 3
assert len(logger2.log) == 2   # only subscribed to login
emitter.unsubscribe("login", logger2)
emitter.emit("login", {"user": "charlie"})
assert len(logger2.log) == 2   # no longer receives login events
print("All tests passed.")
''',
    },

    # ── OOP: Strategy Pattern ─────────────────────────────────────────────────

    {
        "keys": ["strategy pattern", "strategy design pattern", "implement strategy"],
        "lang": "python",
        "title": "Strategy Pattern",
        "complexity": "Strategy swap: O(1)",
        "code": '''\
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: list[int]) -> list[int]: ...

class BubbleSortStrategy(SortStrategy):
    def sort(self, data: list[int]) -> list[int]:
        arr = data[:]
        for i in range(len(arr)):
            for j in range(len(arr) - i - 1):
                if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr

class PythonSortStrategy(SortStrategy):
    def sort(self, data: list[int]) -> list[int]:
        return sorted(data)

class Sorter:
    """Context — delegates sorting to a swappable strategy."""

    def __init__(self, strategy: SortStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: SortStrategy) -> None:
        self._strategy = strategy

    def sort(self, data: list[int]) -> list[int]:
        return self._strategy.sort(data)

# --- Test ---
data = [5, 3, 8, 1, 9, 2]
sorter = Sorter(BubbleSortStrategy())
assert sorter.sort(data) == [1, 2, 3, 5, 8, 9]
sorter.set_strategy(PythonSortStrategy())
assert sorter.sort(data) == [1, 2, 3, 5, 8, 9]
print("All tests passed.")
''',
    },

    # ── Functional: Map / Filter / Reduce ────────────────────────────────────

    {
        "keys": ["map filter reduce", "functional programming", "higher order functions",
                 "map filter python"],
        "lang": "python",
        "title": "Map, Filter, Reduce (Functional Programming)",
        "complexity": "O(n) each",
        "code": '''\
from functools import reduce
from typing import TypeVar, Callable, Iterable

T = TypeVar("T")
U = TypeVar("U")

def my_map(func: Callable[[T], U], iterable: Iterable[T]) -> list[U]:
    return [func(x) for x in iterable]

def my_filter(pred: Callable[[T], bool], iterable: Iterable[T]) -> list[T]:
    return [x for x in iterable if pred(x)]

def my_reduce(func: Callable[[U, T], U], iterable: Iterable[T], initial: U) -> U:
    result = initial
    for x in iterable:
        result = func(result, x)
    return result

# Compose: pipeline of transformations
def pipeline(*funcs: Callable) -> Callable:
    """Apply functions left-to-right."""
    return lambda x: reduce(lambda v, f: f(v), funcs, x)

# --- Test ---
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
assert my_map(lambda x: x * 2, nums)          == [2,4,6,8,10,12,14,16,18,20]
assert my_filter(lambda x: x % 2 == 0, nums)  == [2,4,6,8,10]
assert my_reduce(lambda acc, x: acc + x, nums, 0) == 55

double_evens = pipeline(
    lambda xs: my_filter(lambda x: x % 2 == 0, xs),
    lambda xs: my_map(lambda x: x * 2, xs),
)
assert double_evens([1,2,3,4,5]) == [4, 8]
print("All tests passed.")
''',
    },

    # ── Functional: Memoization Decorator ────────────────────────────────────

    {
        "keys": ["memoization", "memoize decorator", "memoize function", "cache decorator"],
        "lang": "python",
        "title": "Memoization Decorator",
        "complexity": "First call: O(f(n)) | Cached call: O(1)",
        "code": '''\
import functools
import time
from typing import Callable, Any

def memoize(func: Callable) -> Callable:
    """Cache results of pure functions by their arguments."""
    cache: dict[tuple, Any] = {}

    @functools.wraps(func)
    def wrapper(*args: Any) -> Any:
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    wrapper.cache = cache       # type: ignore[attr-defined]
    wrapper.cache_clear = lambda: cache.clear()  # type: ignore[attr-defined]
    return wrapper

@memoize
def fib(n: int) -> int:
    if n <= 1: return n
    return fib(n - 1) + fib(n - 2)

# --- Test ---
assert fib(0) == 0
assert fib(10) == 55
assert fib(30) == 832040
assert 30 in [k[0] for k in fib.cache]
fib.cache_clear()
assert fib.cache == {}
print("All tests passed.")
''',
    },

    # ── Functional: Currying ──────────────────────────────────────────────────

    {
        "keys": ["currying", "curry function", "curried function", "partial application currying"],
        "lang": "python",
        "title": "Currying and Partial Application",
        "complexity": "O(1) per application",
        "code": '''\
import functools
from typing import Callable, Any

def curry(func: Callable) -> Callable:
    """Transform f(a, b, c) into f(a)(b)(c). Works for any arity."""
    import inspect
    arity = len(inspect.signature(func).parameters)

    @functools.wraps(func)
    def curried(*args: Any) -> Any:
        if len(args) >= arity:
            return func(*args[:arity])
        return lambda *more: curried(*(args + more))

    return curried

def compose(*funcs: Callable) -> Callable:
    """Right-to-left function composition: compose(f, g)(x) = f(g(x))."""
    return functools.reduce(lambda f, g: lambda *args: f(g(*args)), funcs)

# --- Test ---
@curry
def add(a: int, b: int, c: int) -> int:
    return a + b + c

add5 = add(2)(3)    # partial application
assert add5(4) == 9
assert add(1)(2)(3) == 6
assert add(1, 2)(3) == 6
assert add(1, 2, 3) == 6

double = lambda x: x * 2
inc    = lambda x: x + 1
double_then_inc = compose(inc, double)
assert double_then_inc(5) == 11   # (5*2)+1
print("All tests passed.")
''',
    },

    # ── File I/O: JSON Read/Write ─────────────────────────────────────────────

    {
        "keys": ["json read", "json write", "read json file", "write json file",
                 "json file python"],
        "lang": "python",
        "title": "JSON Read / Write",
        "complexity": "O(n) where n = JSON size",
        "code": '''\
import json
import os
import tempfile
from typing import Any

def write_json(data: Any, path: str, indent: int = 2) -> None:
    """Write Python object to a JSON file (UTF-8, with atomic write)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    os.replace(tmp, path)   # atomic on POSIX; best-effort on Windows

def read_json(path: str) -> Any:
    """Read JSON file and return Python object."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def json_to_str(data: Any, indent: int = 2) -> str:
    return json.dumps(data, indent=indent, ensure_ascii=False)

def str_to_json(s: str) -> Any:
    return json.loads(s)

# --- Test ---
payload = {"name": "Alice", "scores": [10, 20, 30], "active": True}

with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
    path = tmp.name

write_json(payload, path)
loaded = read_json(path)
assert loaded == payload
assert loaded["scores"] == [10, 20, 30]
os.unlink(path)

assert str_to_json(json_to_str(payload)) == payload
print("All tests passed.")
''',
    },

    # ── Security: Base64 ─────────────────────────────────────────────────────

    {
        "keys": ["base64", "base64 encode", "base64 decode", "base 64 encoding"],
        "lang": "python",
        "title": "Base64 Encode / Decode",
        "complexity": "O(n)",
        "code": '''\
import base64

def b64_encode(data: str | bytes, encoding: str = "utf-8") -> str:
    """Encode string or bytes to Base64 string."""
    if isinstance(data, str):
        data = data.encode(encoding)
    return base64.b64encode(data).decode("ascii")

def b64_decode(data: str, encoding: str = "utf-8") -> str:
    """Decode Base64 string back to original string."""
    return base64.b64decode(data.encode("ascii")).decode(encoding)

def b64_encode_url_safe(data: str) -> str:
    """URL-safe Base64 (uses - and _ instead of + and /)."""
    return base64.urlsafe_b64encode(data.encode()).decode("ascii")

def b64_decode_url_safe(data: str) -> str:
    return base64.urlsafe_b64decode(data.encode("ascii")).decode()

# --- Test ---
msg = "Hello, World! 🌍"
encoded = b64_encode(msg)
assert b64_decode(encoded) == msg

url_enc = b64_encode_url_safe("safe+test/string")
assert "+" not in url_enc and "/" not in url_enc
assert b64_decode_url_safe(url_enc) == "safe+test/string"
print("All tests passed.")
''',
    },

    # ── Security: SHA256 ─────────────────────────────────────────────────────

    {
        "keys": ["sha256", "sha 256", "hash sha256", "sha256 hash", "sha2 hash"],
        "lang": "python",
        "title": "SHA-256 Hashing",
        "complexity": "O(n) where n = input length",
        "code": '''\
import hashlib
import hmac as _hmac
import os

def sha256(data: str | bytes, encoding: str = "utf-8") -> str:
    """Return lowercase hex SHA-256 digest."""
    if isinstance(data, str):
        data = data.encode(encoding)
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: str) -> str:
    """Compute SHA-256 of a file in chunks (memory-efficient)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def hash_password(password: str) -> tuple[str, str]:
    """Hash a password with a random salt. Returns (hash_hex, salt_hex)."""
    salt = os.urandom(32)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return dk.hex(), salt.hex()

def verify_password(password: str, hash_hex: str, salt_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return dk.hex() == hash_hex

# --- Test ---
assert sha256("hello") == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
assert sha256(b"hello") == sha256("hello")
assert sha256("") != sha256("a")

h, s = hash_password("mypassword")
assert verify_password("mypassword", h, s)
assert not verify_password("wrongpass", h, s)
print("All tests passed.")
''',
    },

    # ── Data Processing: Statistics ───────────────────────────────────────────

    {
        "keys": ["mean median mode", "statistics python", "calculate mean", "standard deviation",
                 "descriptive statistics"],
        "lang": "python",
        "title": "Descriptive Statistics (Mean, Median, Mode, Std Dev)",
        "complexity": "O(n) for mean/std | O(n log n) for median",
        "code": '''\
from collections import Counter
import math

def mean(data: list[float]) -> float:
    if not data: raise ValueError("empty list")
    return sum(data) / len(data)

def median(data: list[float]) -> float:
    if not data: raise ValueError("empty list")
    s = sorted(data)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2

def mode(data: list) -> list:
    """Return most common values (may be multiple if tied)."""
    counts = Counter(data)
    max_count = max(counts.values())
    return [k for k, v in counts.items() if v == max_count]

def variance(data: list[float], population: bool = True) -> float:
    m = mean(data)
    n = len(data) if population else len(data) - 1
    if n == 0: raise ValueError("insufficient data")
    return sum((x - m) ** 2 for x in data) / n

def std_dev(data: list[float], population: bool = True) -> float:
    return math.sqrt(variance(data, population))

def summary(data: list[float]) -> dict:
    return {
        "n":      len(data),
        "mean":   mean(data),
        "median": median(data),
        "mode":   mode(data),
        "std":    std_dev(data),
        "min":    min(data),
        "max":    max(data),
        "range":  max(data) - min(data),
    }

# --- Test ---
data = [2, 4, 4, 4, 5, 5, 7, 9]
assert mean(data)   == 5.0
assert median(data) == 4.5
assert mode(data)   == [4]
assert abs(std_dev(data) - 2.0) < 1e-9
s = summary(data)
assert s["min"] == 2 and s["max"] == 9
print("All tests passed.")
''',
    },

    # ── Data Processing: Moving Average ──────────────────────────────────────

    {
        "keys": ["moving average", "rolling average", "sliding average", "window average"],
        "lang": "python",
        "title": "Moving Average (Simple and Exponential)",
        "complexity": "SMA: O(n) | EMA: O(n)",
        "code": '''\
from collections import deque

def simple_moving_average(data: list[float], window: int) -> list[float]:
    """Compute SMA with a fixed window. Returns list shorter by (window-1)."""
    if window > len(data):
        return []
    result: list[float] = []
    window_sum = sum(data[:window])
    result.append(window_sum / window)
    for i in range(window, len(data)):
        window_sum += data[i] - data[i - window]
        result.append(window_sum / window)
    return result

def exponential_moving_average(data: list[float], alpha: float = 0.3) -> list[float]:
    """EMA with smoothing factor alpha. More weight on recent values."""
    if not data: return []
    ema = [data[0]]
    for x in data[1:]:
        ema.append(alpha * x + (1 - alpha) * ema[-1])
    return ema

# --- Test ---
prices = [10, 11, 12, 11, 10, 11, 12, 13, 12, 11]
sma3 = simple_moving_average(prices, 3)
assert len(sma3) == 8
assert abs(sma3[0] - 11.0) < 1e-9   # (10+11+12)/3
assert abs(sma3[1] - (11+12+11)/3) < 1e-9

ema = exponential_moving_average(prices, alpha=0.5)
assert len(ema) == len(prices)
assert ema[0] == prices[0]
print("All tests passed.")
''',
    },

    # ── Concurrency: Thread Pool ──────────────────────────────────────────────

    {
        "keys": ["thread pool", "threadpool", "concurrent futures", "parallel tasks"],
        "lang": "python",
        "title": "Thread Pool (concurrent.futures)",
        "complexity": "Scales with I/O concurrency",
        "code": '''\
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Callable, TypeVar, Any

T = TypeVar("T")

def run_parallel(func: Callable, items: list, max_workers: int = 4) -> list:
    """Run func(item) in parallel for each item. Returns results in order."""
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(func, item): i for i, item in enumerate(items)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return results

def run_parallel_with_errors(func: Callable, items: list, max_workers: int = 4) -> list[dict]:
    """Return list of {index, result, error} dicts."""
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(func, item): i for i, item in enumerate(items)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = {"index": idx, "result": future.result(), "error": None}
            except Exception as e:
                results[idx] = {"index": idx, "result": None, "error": str(e)}
    return results

# --- Test ---
def slow_square(x: int) -> int:
    time.sleep(0.01)
    return x * x

nums = [1, 2, 3, 4, 5]
results = run_parallel(slow_square, nums)
assert results == [1, 4, 9, 16, 25]

def sometimes_fails(x: int) -> int:
    if x == 3: raise ValueError("bad value")
    return x

err_results = run_parallel_with_errors(sometimes_fails, [1, 2, 3, 4])
assert err_results[0]["result"] == 1
assert err_results[2]["error"] == "bad value"
print("All tests passed.")
''',
    },

    # ── Concurrency: Producer-Consumer ────────────────────────────────────────

    {
        "keys": ["producer consumer", "producer consumer pattern", "queue threading"],
        "lang": "python",
        "title": "Producer-Consumer Pattern",
        "complexity": "O(1) per put/get (bounded queue)",
        "code": '''\
import queue
import threading
import time

def producer_consumer_demo() -> list[int]:
    """Producer puts items; consumer processes them. Synchronised via queue."""
    work_queue: queue.Queue[int] = queue.Queue(maxsize=10)
    results: list[int] = []
    lock = threading.Lock()

    def producer(n: int) -> None:
        for i in range(n):
            work_queue.put(i)
        work_queue.put(None)   # sentinel: signals end of work

    def consumer() -> None:
        while True:
            item = work_queue.get()
            if item is None:
                break
            with lock:
                results.append(item * 2)
            work_queue.task_done()

    t_prod = threading.Thread(target=producer, args=(5,))
    t_cons = threading.Thread(target=consumer)
    t_prod.start(); t_cons.start()
    t_prod.join();  t_cons.join()
    return sorted(results)

# --- Test ---
result = producer_consumer_demo()
assert result == [0, 2, 4, 6, 8]
print("All tests passed.")
''',
    },

    # ── Recursion: Tower of Hanoi ─────────────────────────────────────────────

    {
        "keys": ["tower of hanoi", "hanoi", "hanoi problem", "towers of hanoi"],
        "lang": "python",
        "title": "Tower of Hanoi",
        "complexity": "Time: O(2^n) | Moves: 2^n - 1",
        "code": '''\
def hanoi(n: int, source: str = "A", target: str = "C", aux: str = "B") -> list[tuple[str, str]]:
    """Return list of (from, to) moves to solve n-disk Tower of Hanoi.

    Strategy: move n-1 disks to aux, move largest to target, move n-1 to target.
    """
    moves: list[tuple[str, str]] = []
    def _solve(disks: int, src: str, tgt: str, aux: str) -> None:
        if disks == 0:
            return
        _solve(disks - 1, src, aux, tgt)
        moves.append((src, tgt))
        _solve(disks - 1, aux, tgt, src)
    _solve(n, source, target, aux)
    return moves

# --- Test ---
moves = hanoi(3)
assert len(moves) == 7          # 2^3 - 1
assert moves[0]  == ("A", "C")  # first move
assert moves[-1] == ("A", "C")  # last move
# Verify: simulate pegs
pegs: dict[str, list[int]] = {"A": [3,2,1], "B": [], "C": []}
for src, tgt in moves:
    pegs[tgt].append(pegs[src].pop())
assert pegs["A"] == [] and pegs["C"] == [3, 2, 1]
print("All tests passed.")
''',
    },

    # ── Recursion/Backtracking: N-Queens ──────────────────────────────────────

    {
        "keys": ["n queens", "n-queens", "eight queens", "queens problem", "n queens problem"],
        "lang": "python",
        "title": "N-Queens Problem",
        "complexity": "Time: O(n!) worst | Backtracking prunes heavily",
        "code": '''\
def solve_n_queens(n: int) -> list[list[str]]:
    """Return all valid n-queens board configurations.

    Each solution is a list of n strings, each of length n.
    Q = queen, . = empty.
    """
    solutions: list[list[str]] = []
    cols:    set[int] = set()
    diag1:   set[int] = set()   # row - col
    diag2:   set[int] = set()   # row + col
    board    = [["." ] * n for _ in range(n)]

    def backtrack(row: int) -> None:
        if row == n:
            solutions.append(["".join(r) for r in board])
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            cols.add(col); diag1.add(row - col); diag2.add(row + col)
            board[row][col] = "Q"
            backtrack(row + 1)
            board[row][col] = "."
            cols.discard(col); diag1.discard(row - col); diag2.discard(row + col)

    backtrack(0)
    return solutions

# --- Test ---
assert len(solve_n_queens(4)) == 2
assert len(solve_n_queens(8)) == 92
sol = solve_n_queens(4)[0]
assert len(sol) == 4
assert all(row.count("Q") == 1 for row in sol)
print("All tests passed.")
''',
    },

    # ── Recursion/Backtracking: Combinations ─────────────────────────────────

    {
        "keys": ["combinations", "power set", "all subsets", "combination algorithm",
                 "generate subsets"],
        "lang": "python",
        "title": "Combinations and Power Set",
        "complexity": "Power set: O(2^n) | k-combinations: O(C(n,k))",
        "code": '''\
from typing import TypeVar

T = TypeVar("T")

def power_set(items: list[T]) -> list[list[T]]:
    """Return all 2^n subsets of items."""
    result: list[list[T]] = [[]]
    for item in items:
        result += [subset + [item] for subset in result]
    return result

def combinations(items: list[T], k: int) -> list[list[T]]:
    """Return all C(n,k) combinations of size k."""
    if k == 0:
        return [[]]
    if not items or k > len(items):
        return []
    result: list[list[T]] = []
    def backtrack(start: int, path: list[T]) -> None:
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(start, len(items)):
            path.append(items[i])
            backtrack(i + 1, path)
            path.pop()
    backtrack(0, [])
    return result

# --- Test ---
ps = power_set([1, 2, 3])
assert len(ps) == 8
assert [] in ps
assert [1, 2, 3] in ps

c23 = combinations([1, 2, 3, 4], 2)
assert len(c23) == 6
assert [1, 2] in c23 and [3, 4] in c23
print("All tests passed.")
''',
    },

    # ── Interview: Valid Parentheses ──────────────────────────────────────────

    {
        "keys": ["valid parentheses", "balanced brackets", "matching brackets",
                 "balanced parentheses"],
        "lang": "python",
        "title": "Valid Parentheses / Balanced Brackets",
        "complexity": "Time: O(n) | Space: O(n)",
        "code": '''\
def is_valid_parentheses(s: str) -> bool:
    """Return True if brackets in s are correctly matched and nested.

    Handles (), [], {}.
    """
    stack: list[str] = []
    matching = {")": "(", "]": "[", "}": "{"}
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != matching[ch]:
                return False
            stack.pop()
    return len(stack) == 0

# --- Test ---
assert is_valid_parentheses("()")         == True
assert is_valid_parentheses("()[]{}")     == True
assert is_valid_parentheses("{[()]}")     == True
assert is_valid_parentheses("(]")         == False
assert is_valid_parentheses("([)]")       == False
assert is_valid_parentheses("{[]}")       == True
assert is_valid_parentheses("")           == True
assert is_valid_parentheses("((")         == False
print("All tests passed.")
''',
    },

    # ── Interview: Merge Intervals ────────────────────────────────────────────

    {
        "keys": ["merge intervals", "overlapping intervals", "interval merging"],
        "lang": "python",
        "title": "Merge Overlapping Intervals",
        "complexity": "Time: O(n log n) | Space: O(n)",
        "code": '''\
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """Merge all overlapping intervals. Input: [[start, end], ...]."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0][:]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:       # overlaps
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged

# --- Test ---
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([[1,4],[4,5]])                == [[1,5]]
assert merge_intervals([[1,4],[2,3]])                == [[1,4]]
assert merge_intervals([])                           == []
print("All tests passed.")
''',
    },

    # ── Interview: Product Except Self ────────────────────────────────────────

    {
        "keys": ["product except self", "product of array except self", "array product"],
        "lang": "python",
        "title": "Product of Array Except Self",
        "complexity": "Time: O(n) | Space: O(1) output only",
        "code": '''\
def product_except_self(nums: list[int]) -> list[int]:
    """Return output[i] = product of all elements except nums[i].

    No division. Two-pass: left products, then right products.
    """
    n = len(nums)
    result = [1] * n
    left = 1
    for i in range(n):
        result[i] = left
        left *= nums[i]
    right = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right
        right *= nums[i]
    return result

# --- Test ---
assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
assert product_except_self([0, 1, 2])    == [2, 0, 0]
assert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]
print("All tests passed.")
''',
    },

    # ── Interview: Sliding Window Maximum ────────────────────────────────────

    {
        "keys": ["sliding window max", "sliding window maximum", "window max deque"],
        "lang": "python",
        "title": "Sliding Window Maximum",
        "complexity": "Time: O(n) | Space: O(k)",
        "code": '''\
from collections import deque

def sliding_window_max(nums: list[int], k: int) -> list[int]:
    """Return max of each sliding window of size k using a monotonic deque.

    Deque stores indices; front is always the index of the current window max.
    """
    result: list[int] = []
    dq: deque[int] = deque()   # indices, decreasing order of nums values

    for i, x in enumerate(nums):
        # remove indices outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        # remove smaller values from back (they can never be max)
        while dq and nums[dq[-1]] < x:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result

# --- Test ---
assert sliding_window_max([1,3,-1,-3,5,3,6,7], 3) == [3,3,5,5,6,7]
assert sliding_window_max([1], 1)                  == [1]
assert sliding_window_max([4,3,2,1], 2)            == [4,3,2]
print("All tests passed.")
''',
    },

    # ── Interview: Dutch National Flag ────────────────────────────────────────

    {
        "keys": ["dutch national flag", "three way partition", "sort 0 1 2",
                 "three color sort"],
        "lang": "python",
        "title": "Dutch National Flag (3-Way Partition)",
        "complexity": "Time: O(n) | Space: O(1)",
        "code": '''\
def dutch_national_flag(arr: list[int]) -> list[int]:
    """Sort array of 0s, 1s, 2s in one pass (Dijkstra\'s 3-way partition).

    Uses three pointers: lo (0-boundary), mid (current), hi (2-boundary).
    """
    arr = arr[:]
    lo = mid = 0
    hi = len(arr) - 1
    while mid <= hi:
        if arr[mid] == 0:
            arr[lo], arr[mid] = arr[mid], arr[lo]
            lo += 1; mid += 1
        elif arr[mid] == 1:
            mid += 1
        else:
            arr[mid], arr[hi] = arr[hi], arr[mid]
            hi -= 1
    return arr

# --- Test ---
assert dutch_national_flag([2,0,2,1,1,0])       == [0,0,1,1,2,2]
assert dutch_national_flag([2,0,1])              == [0,1,2]
assert dutch_national_flag([0])                  == [0]
assert dutch_national_flag([1,1,1,0,0,0,2,2,2]) == [0,0,0,1,1,1,2,2,2]
print("All tests passed.")
''',
    },

    # ── Interview: Rotate Array ───────────────────────────────────────────────

    {
        "keys": ["rotate array", "array rotation", "rotate list"],
        "lang": "python",
        "title": "Rotate Array",
        "complexity": "Time: O(n) | Space: O(1) in-place",
        "code": '''\
def rotate_array(nums: list[int], k: int) -> list[int]:
    """Rotate array right by k steps in-place using reversal trick."""
    n = len(nums)
    if n == 0: return nums
    nums = nums[:]
    k %= n
    nums.reverse()
    nums[:k] = nums[:k][::-1]
    nums[k:]  = nums[k:][::-1]
    return nums

def rotate_left(nums: list[int], k: int) -> list[int]:
    n = len(nums)
    if n == 0: return nums
    k %= n
    return nums[k:] + nums[:k]

# --- Test ---
assert rotate_array([1,2,3,4,5,6,7], 3) == [5,6,7,1,2,3,4]
assert rotate_array([1,2], 3)           == [2,1]   # k > n
assert rotate_left([1,2,3,4,5], 2)      == [3,4,5,1,2]
print("All tests passed.")
''',
    },

    # ── SQL: CRUD Operations ──────────────────────────────────────────────────

    {
        "keys": ["crud sql", "sql crud", "insert update delete select", "basic sql operations"],
        "lang": "sql",
        "title": "SQL CRUD Operations",
        "complexity": "Depends on indexes",
        "code": '''\
-- CREATE TABLE
CREATE TABLE users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT    NOT NULL UNIQUE,
    email      TEXT    NOT NULL,
    age        INTEGER CHECK(age >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- INSERT (Create)
INSERT INTO users (username, email, age) VALUES
    (\'alice\', \'alice@example.com\', 30),
    (\'bob\',   \'bob@example.com\',   25);

-- SELECT (Read)
SELECT id, username, email FROM users WHERE age > 20 ORDER BY username;

-- UPDATE
UPDATE users SET email = \'newemail@example.com\', age = 31
WHERE username = \'alice\';

-- DELETE
DELETE FROM users WHERE username = \'bob\';

-- UPSERT (INSERT or UPDATE if exists) — SQLite syntax
INSERT INTO users (username, email, age)
VALUES (\'charlie\', \'charlie@example.com\', 28)
ON CONFLICT(username) DO UPDATE SET
    email = excluded.email,
    age   = excluded.age;

-- SELECT with JOIN
SELECT u.username, o.order_date, o.total
FROM users u
INNER JOIN orders o ON o.user_id = u.id
WHERE o.total > 100
ORDER BY o.order_date DESC
LIMIT 10;
''',
    },

    # ── SQL: Aggregations ─────────────────────────────────────────────────────

    {
        "keys": ["sql aggregation", "sql aggregate", "group by having", "count sum avg sql"],
        "lang": "sql",
        "title": "SQL Aggregations (GROUP BY, HAVING, COUNT, SUM, AVG)",
        "complexity": "O(n log n) with GROUP BY",
        "code": '''\
-- Basic aggregations
SELECT
    department,
    COUNT(*)         AS employee_count,
    AVG(salary)      AS avg_salary,
    MAX(salary)      AS max_salary,
    MIN(salary)      AS min_salary,
    SUM(salary)      AS total_payroll
FROM employees
GROUP BY department
HAVING COUNT(*) > 5           -- filter AFTER aggregation
ORDER BY avg_salary DESC;

-- Count distinct values
SELECT COUNT(DISTINCT country) AS unique_countries FROM customers;

-- Aggregate with CASE (conditional count)
SELECT
    department,
    COUNT(*) AS total,
    SUM(CASE WHEN salary > 80000 THEN 1 ELSE 0 END) AS high_earners,
    ROUND(100.0 * SUM(CASE WHEN salary > 80000 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_high
FROM employees
GROUP BY department;

-- Running total with window function
SELECT
    order_date,
    amount,
    SUM(amount) OVER (ORDER BY order_date) AS running_total
FROM orders;
''',
    },

    # ── Unique Software: Blockchain ──────────────────────────────────────────

    {
        "keys": ["blockchain", "blockchain ledger", "proof of work", "build blockchain",
                 "implement blockchain"],
        "lang": "python",
        "title": "Blockchain Ledger with Proof of Work",
        "complexity": "Mining: O(difficulty * hash_attempts) | Validation: O(n)",
        "code": '''\
import hashlib
import time
import json
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Block:
    index:      int
    timestamp:  float
    data:       dict
    prev_hash:  str
    nonce:      int = 0
    hash:       str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        content = json.dumps({
            "index":     self.index,
            "timestamp": self.timestamp,
            "data":      self.data,
            "prev_hash": self.prev_hash,
            "nonce":     self.nonce,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

class Blockchain:
    """Append-only chain secured by proof-of-work mining.

    Difficulty = number of leading zeros required in block hash.
    Higher difficulty exponentially increases mining time.
    """

    DIFFICULTY = 3   # 3 leading zeros — keeps demo fast

    def __init__(self) -> None:
        self.chain: list[Block] = []
        self._add_genesis()

    def _add_genesis(self) -> None:
        genesis = Block(0, time.time(), {"genesis": True}, "0" * 64)
        self.chain.append(genesis)

    def mine_block(self, data: dict) -> Block:
        """Mine a new block: increment nonce until hash meets difficulty target."""
        prev = self.chain[-1]
        block = Block(len(self.chain), time.time(), data, prev.hash)
        target = "0" * self.DIFFICULTY
        while not block.hash.startswith(target):
            block.nonce += 1
            block.hash = block.compute_hash()
        self.chain.append(block)
        return block

    def is_valid(self) -> bool:
        """Verify the integrity of the entire chain."""
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i - 1]
            if curr.hash != curr.compute_hash():
                return False   # block was tampered
            if curr.prev_hash != prev.hash:
                return False   # chain link broken
        return True

    def __len__(self) -> int:
        return len(self.chain)

# --- Test ---
bc = Blockchain()
b1 = bc.mine_block({"tx": "Alice -> Bob: 10 coins"})
b2 = bc.mine_block({"tx": "Bob -> Charlie: 5 coins"})

assert len(bc) == 3                             # genesis + 2
assert bc.is_valid()
assert b1.hash.startswith("0" * Blockchain.DIFFICULTY)
assert b1.prev_hash == bc.chain[0].hash        # links correctly

# Tamper detection
bc.chain[1].data = {"tx": "Alice -> Bob: 9999 coins"}
assert bc.is_valid() == False                  # detects tampering
print("All tests passed.")
''',
    },

    # ── Unique Software: Neural Network ──────────────────────────────────────

    {
        "keys": ["neural network from scratch", "neural network no libraries", "implement neural network",
                 "backpropagation from scratch", "build neural network"],
        "lang": "python",
        "title": "Neural Network from Scratch (No Libraries)",
        "complexity": "Forward/Backward pass: O(layers * neurons²)",
        "code": '''\
import math
import random

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

def sigmoid_deriv(x: float) -> float:
    s = sigmoid(x)
    return s * (1.0 - s)

class NeuralNetwork:
    """Two-layer (one hidden) neural network trained with backpropagation.

    Architecture: input_size -> hidden_size -> output_size
    Activation: sigmoid throughout.
    Loss: mean squared error.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 lr: float = 0.5) -> None:
        self.lr = lr
        rnd = lambda: random.uniform(-1, 1)
        self.W1 = [[rnd() for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [rnd() for _ in range(hidden_size)]
        self.W2 = [[rnd() for _ in range(hidden_size)] for _ in range(output_size)]
        self.b2 = [rnd() for _ in range(output_size)]

    def _dot(self, weights: list[list[float]], inputs: list[float],
             biases: list[float]) -> list[float]:
        return [sum(w * x for w, x in zip(row, inputs)) + b
                for row, b in zip(weights, biases)]

    def forward(self, x: list[float]) -> tuple[list[float], list[float], list[float], list[float]]:
        z1 = self._dot(self.W1, x, self.b1)
        a1 = [sigmoid(v) for v in z1]
        z2 = self._dot(self.W2, a1, self.b2)
        a2 = [sigmoid(v) for v in z2]
        return z1, a1, z2, a2

    def train(self, X: list[list[float]], Y: list[list[float]], epochs: int = 1000) -> None:
        for _ in range(epochs):
            for x, y in zip(X, Y):
                z1, a1, z2, a2 = self.forward(x)
                # Output layer delta
                delta2 = [(a - t) * sigmoid_deriv(z)
                          for a, t, z in zip(a2, y, z2)]
                # Hidden layer delta
                delta1 = [sum(self.W2[k][j] * delta2[k] for k in range(len(delta2)))
                          * sigmoid_deriv(z1[j])
                          for j in range(len(z1))]
                # Update W2, b2
                for k in range(len(self.W2)):
                    for j in range(len(a1)):
                        self.W2[k][j] -= self.lr * delta2[k] * a1[j]
                    self.b2[k] -= self.lr * delta2[k]
                # Update W1, b1
                for j in range(len(self.W1)):
                    for i in range(len(x)):
                        self.W1[j][i] -= self.lr * delta1[j] * x[i]
                    self.b1[j] -= self.lr * delta1[j]

    def predict(self, x: list[float]) -> list[float]:
        return self.forward(x)[3]

# --- Test: learn XOR ---
random.seed(42)
nn = NeuralNetwork(2, 4, 1, lr=0.5)
X = [[0,0],[0,1],[1,0],[1,1]]
Y = [[0],[1],[1],[0]]
nn.train(X, Y, epochs=5000)
for x, y in zip(X, Y):
    pred = nn.predict(x)[0]
    assert abs(pred - y[0]) < 0.1, f"XOR({x}) failed: got {pred:.3f}, expected {y[0]}"
print("All tests passed. XOR learned successfully.")
''',
    },

    # ── Unique Software: URL Shortener ────────────────────────────────────────

    {
        "keys": ["url shortener", "url shortener service", "shorten url", "link shortener"],
        "lang": "python",
        "title": "URL Shortener Service (In-Memory)",
        "complexity": "Encode/Decode: O(1) | Collision probability: negligible at 6 chars base62",
        "code": '''\
import random
import string
import time
from dataclasses import dataclass, field

_CHARS = string.ascii_letters + string.digits  # base-62

@dataclass
class URLRecord:
    original:   str
    short_code: str
    created_at: float = field(default_factory=time.time)
    hits:       int = 0

class URLShortener:
    """In-memory URL shortener with collision handling and analytics.

    Short codes are 6-character base-62 strings (~56 billion combinations).
    """

    def __init__(self, base_url: str = "https://sh.rt/", code_length: int = 6) -> None:
        self.base_url    = base_url
        self.code_length = code_length
        self._codes:  dict[str, URLRecord] = {}   # short_code -> record
        self._urls:   dict[str, str]       = {}   # original -> short_code

    def shorten(self, url: str) -> str:
        """Return short URL for the given long URL. Idempotent for same URL."""
        if url in self._urls:
            return self.base_url + self._urls[url]
        code = self._generate_code()
        record = URLRecord(original=url, short_code=code)
        self._codes[code] = record
        self._urls[url]   = code
        return self.base_url + code

    def resolve(self, short_url: str) -> str | None:
        """Resolve short URL to original. Returns None if not found."""
        code = short_url.removeprefix(self.base_url)
        record = self._codes.get(code)
        if record:
            record.hits += 1
            return record.original
        return None

    def stats(self, short_url: str) -> dict | None:
        code = short_url.removeprefix(self.base_url)
        rec = self._codes.get(code)
        if not rec: return None
        return {"short": self.base_url + code, "original": rec.original,
                "hits": rec.hits, "created_at": rec.created_at}

    def _generate_code(self) -> str:
        while True:
            code = "".join(random.choices(_CHARS, k=self.code_length))
            if code not in self._codes:
                return code

# --- Test ---
svc = URLShortener()
short1 = svc.shorten("https://example.com/some/very/long/path?query=1")
short2 = svc.shorten("https://github.com/another/long/url")

assert short1.startswith("https://sh.rt/")
assert len(short1) == len("https://sh.rt/") + 6
assert svc.shorten("https://example.com/some/very/long/path?query=1") == short1  # idempotent

resolved = svc.resolve(short1)
assert resolved == "https://example.com/some/very/long/path?query=1"

stats = svc.stats(short1)
assert stats["hits"] == 1

assert svc.resolve("https://sh.rt/xxxxxx") is None
print("All tests passed.")
''',
    },

    # ── Unique Software: Sudoku Solver ────────────────────────────────────────

    {
        "keys": ["sudoku solver", "solve sudoku", "sudoku backtracking", "sudoku algorithm"],
        "lang": "python",
        "title": "Sudoku Solver (Backtracking + Constraint Propagation)",
        "complexity": "Worst case: O(9^81) | Constraint pruning makes practical cases fast",
        "code": '''\
from typing import Optional

Board = list[list[int]]   # 9x9 grid; 0 = empty

def solve_sudoku(board: Board) -> Optional[Board]:
    """Solve 9x9 Sudoku using backtracking with MRV (fewest options first).

    Returns solved board or None if unsolvable.
    """
    import copy
    board = copy.deepcopy(board)
    if _solve(board):
        return board
    return None

def _solve(board: Board) -> bool:
    cell = _find_empty(board)
    if cell is None:
        return True   # solved
    row, col = cell
    for num in _candidates(board, row, col):
        board[row][col] = num
        if _solve(board):
            return True
        board[row][col] = 0   # backtrack
    return False

def _find_empty(board: Board) -> Optional[tuple[int, int]]:
    """Find empty cell with fewest valid candidates (MRV heuristic)."""
    best = None
    best_count = 10
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                n = len(_candidates(board, r, c))
                if n < best_count:
                    best_count, best = n, (r, c)
    return best

def _candidates(board: Board, row: int, col: int) -> set[int]:
    used: set[int] = set()
    used.update(board[row])
    used.update(board[r][col] for r in range(9))
    br, bc = (row // 3) * 3, (col // 3) * 3
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            used.add(board[r][c])
    return set(range(1, 10)) - used

def is_valid_solution(board: Board) -> bool:
    expected = set(range(1, 10))
    for r in range(9):
        if set(board[r]) != expected: return False
    for c in range(9):
        if {board[r][c] for r in range(9)} != expected: return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = {board[br+dr][bc+dc] for dr in range(3) for dc in range(3)}
            if box != expected: return False
    return True

# --- Test ---
puzzle = [
    [5,3,0, 0,7,0, 0,0,0],
    [6,0,0, 1,9,5, 0,0,0],
    [0,9,8, 0,0,0, 0,6,0],
    [8,0,0, 0,6,0, 0,0,3],
    [4,0,0, 8,0,3, 0,0,1],
    [7,0,0, 0,2,0, 0,0,6],
    [0,6,0, 0,0,0, 2,8,0],
    [0,0,0, 4,1,9, 0,0,5],
    [0,0,0, 0,8,0, 0,7,9],
]
solution = solve_sudoku(puzzle)
assert solution is not None
assert is_valid_solution(solution)
print("All tests passed.")
''',
    },

    # ── Unique Software: Pub/Sub Event Bus ────────────────────────────────────

    {
        "keys": ["pub sub", "event bus", "publish subscribe", "pubsub", "message bus"],
        "lang": "python",
        "title": "Pub/Sub Event Bus",
        "complexity": "Publish: O(n subscribers) | Subscribe: O(1)",
        "code": '''\
from __future__ import annotations
import threading
from typing import Callable, Any
from collections import defaultdict

class EventBus:
    """Thread-safe publish-subscribe event bus.

    Supports wildcard (*) subscriptions, one-time handlers, and async dispatch.
    """

    def __init__(self) -> None:
        self._subs: dict[str, list[Callable]] = defaultdict(list)
        self._once: dict[str, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event: str, handler: Callable[[Any], None]) -> None:
        with self._lock:
            self._subs[event].append(handler)

    def once(self, event: str, handler: Callable[[Any], None]) -> None:
        """Handler fires exactly once then auto-removes."""
        with self._lock:
            self._once[event].append(handler)

    def unsubscribe(self, event: str, handler: Callable) -> None:
        with self._lock:
            self._subs[event] = [h for h in self._subs[event] if h is not handler]

    def publish(self, event: str, data: Any = None) -> int:
        """Publish event. Returns number of handlers called."""
        with self._lock:
            handlers = list(self._subs.get(event, []))
            handlers += list(self._subs.get("*", []))         # wildcard
            once_handlers = list(self._once.pop(event, []))
        for h in handlers + once_handlers:
            h(data)
        return len(handlers) + len(once_handlers)

    def publish_async(self, event: str, data: Any = None) -> threading.Thread:
        """Publish in a background thread — non-blocking."""
        t = threading.Thread(target=self.publish, args=(event, data), daemon=True)
        t.start()
        return t

# --- Test ---
bus = EventBus()
log: list[str] = []

bus.subscribe("user.login",  lambda d: log.append(f"login:{d['user']}"))
bus.subscribe("user.login",  lambda d: log.append(f"audit:{d['user']}"))
bus.subscribe("*",           lambda d: log.append("wildcard"))

fired_once = []
bus.once("user.signup", lambda d: fired_once.append(d))

bus.publish("user.login",  {"user": "alice"})
bus.publish("user.signup", {"user": "bob"})
bus.publish("user.signup", {"user": "charlie"})  # once handler already removed

assert "login:alice"  in log
assert "audit:alice"  in log
assert "wildcard"     in log
assert len(fired_once) == 1            # fired exactly once
assert fired_once[0]["user"] == "bob"
print("All tests passed.")
''',
    },

    # ── Unique Software: Job Scheduler ────────────────────────────────────────

    {
        "keys": ["job scheduler", "task scheduler", "schedule tasks", "cron scheduler",
                 "run tasks at intervals"],
        "lang": "python",
        "title": "Job Scheduler (Run Tasks at Intervals)",
        "complexity": "Schedule: O(log n) heap | Tick: O(k) jobs due",
        "code": '''\
import heapq
import time
import threading
from typing import Callable, Optional

class Job:
    def __init__(self, name: str, func: Callable, interval: float,
                 repeat: bool = True) -> None:
        self.name      = name
        self.func      = func
        self.interval  = interval
        self.repeat    = repeat
        self.next_run  = time.monotonic() + interval
        self.run_count = 0
        self.errors: list[str] = []

    def __lt__(self, other: "Job") -> bool:
        return self.next_run < other.next_run

class Scheduler:
    """Min-heap job scheduler. Jobs run in a background thread."""

    def __init__(self) -> None:
        self._heap:    list[Job] = []
        self._lock     = threading.Lock()
        self._stop     = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def schedule(self, name: str, func: Callable, interval: float,
                 repeat: bool = True) -> Job:
        job = Job(name, func, interval, repeat)
        with self._lock:
            heapq.heappush(self._heap, job)
        return job

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                due = [j for j in self._heap if j.next_run <= now]
                self._heap = [j for j in self._heap if j.next_run > now]
                heapq.heapify(self._heap)
            for job in due:
                try:
                    job.func()
                    job.run_count += 1
                except Exception as e:
                    job.errors.append(str(e))
                if job.repeat:
                    job.next_run = time.monotonic() + job.interval
                    with self._lock:
                        heapq.heappush(self._heap, job)
            time.sleep(0.01)

# --- Test ---
results: list[str] = []
lock = threading.Lock()

sched = Scheduler()
sched.schedule("heartbeat", lambda: results.append("beat"), interval=0.05)
sched.schedule("once",      lambda: results.append("once"), interval=0.02, repeat=False)
sched.start()
time.sleep(0.25)
sched.stop()

assert "beat" in results
assert results.count("once") == 1      # ran exactly once
assert results.count("beat") >= 3      # repeated at least 3 times in 250ms
print("All tests passed.")
''',
    },

    # ── Unique Software: Consistent Hash Ring ─────────────────────────────────

    {
        "keys": ["consistent hash", "hash ring", "consistent hashing", "distributed hash",
                 "virtual nodes hash"],
        "lang": "python",
        "title": "Consistent Hash Ring (Distributed Systems)",
        "complexity": "Add/Remove node: O(v log v) | Lookup: O(log n) where v = virtual nodes",
        "code": '''\
import hashlib
import bisect

class ConsistentHashRing:
    """Consistent hash ring with virtual nodes for even key distribution.

    Used in distributed caches (Redis Cluster) and load balancers to minimise
    key remapping when nodes are added or removed.
    """

    def __init__(self, virtual_nodes: int = 150) -> None:
        self.virtual_nodes = virtual_nodes
        self._ring: dict[int, str] = {}
        self._sorted_keys: list[int] = []

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node: str) -> None:
        for i in range(self.virtual_nodes):
            vkey = self._hash(f"{node}:vnode:{i}")
            self._ring[vkey] = node
            bisect.insort(self._sorted_keys, vkey)

    def remove_node(self, node: str) -> None:
        for i in range(self.virtual_nodes):
            vkey = self._hash(f"{node}:vnode:{i}")
            del self._ring[vkey]
            idx = bisect.bisect_left(self._sorted_keys, vkey)
            self._sorted_keys.pop(idx)

    def get_node(self, key: str) -> str | None:
        if not self._ring:
            return None
        h = self._hash(key)
        idx = bisect.bisect_right(self._sorted_keys, h) % len(self._sorted_keys)
        return self._ring[self._sorted_keys[idx]]

    def distribution(self, keys: list[str]) -> dict[str, int]:
        """Count how many keys map to each node."""
        counts: dict[str, int] = {}
        for k in keys:
            node = self.get_node(k)
            if node:
                counts[node] = counts.get(node, 0) + 1
        return counts

# --- Test ---
ring = ConsistentHashRing(virtual_nodes=100)
nodes = ["cache-01", "cache-02", "cache-03"]
for n in nodes:
    ring.add_node(n)

keys = [f"user:{i}" for i in range(1000)]
dist = ring.distribution(keys)

# Each node should handle roughly 1/3 of keys (within 20% tolerance)
for node, count in dist.items():
    assert 200 < count < 500, f"{node} handles {count}/1000 keys — badly unbalanced"

# Remove one node; existing keys remap minimally
ring.remove_node("cache-03")
dist2 = ring.distribution(keys)
assert "cache-03" not in dist2

# A specific key always maps to the same node (deterministic)
assert ring.get_node("user:42") == ring.get_node("user:42")
print("All tests passed.")
''',
    },

    # ── Async ───────────────────────────────────────────────────────────────

    {
        "keys": ["async http", "async fetch", "asyncio fetch", "aiohttp", "async request",
                 "async http fetch", "async get request"],
        "lang": "python",
        "title": "Async HTTP Fetch (asyncio + urllib)",
        "complexity": "I/O-bound: scales with concurrency, not CPU",
        "code": '''\
import asyncio
import urllib.request
import json
from typing import Any
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=10)

async def async_fetch(url: str, timeout: float = 10.0) -> dict[str, Any]:
    """Async HTTP GET using ThreadPoolExecutor (no third-party deps).

    For production use, prefer aiohttp or httpx which are truly non-blocking.
    This pattern is correct for asyncio apps that must avoid blocking the loop.
    """
    loop = asyncio.get_event_loop()

    def _fetch_sync() -> dict[str, Any]:
        req = urllib.request.Request(url, headers={"User-Agent": "Zophiel/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            content_type = resp.headers.get("Content-Type", "")
            return {
                "status":  resp.status,
                "url":     url,
                "body":    body,
                "json":    json.loads(body) if "json" in content_type else None,
                "headers": dict(resp.headers),
            }

    return await loop.run_in_executor(_executor, _fetch_sync)

async def fetch_many(urls: list[str]) -> list[dict[str, Any]]:
    """Fetch multiple URLs concurrently and return all results."""
    tasks = [async_fetch(url) for url in urls]
    return await asyncio.gather(*tasks, return_exceptions=True)

# --- Test (uses a real public API) ---
async def _test() -> None:
    result = await async_fetch("https://httpbin.org/get")
    assert result["status"] == 200
    assert "url" in result
    print(f"Fetched: {result['url']}  status={result['status']}")
    print("All tests passed.")

# Run: asyncio.run(_test())
# Uncomment below to execute test (requires network access):
# asyncio.run(_test())
print("Async fetch module loaded. Call asyncio.run(_test()) to verify.")
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
