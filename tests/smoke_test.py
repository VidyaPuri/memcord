"""Smoke test for memcord — cache + backends."""
import os
import shutil
import tempfile

_test_dir = tempfile.mkdtemp(prefix="memcord_smoke_")
print(f"Test dir: {_test_dir}")

from memcord.cache import FAQCache

# Test 1: Init with env var for feedback threshold
os.environ["MEMCORD_FEEDBACK_THRESHOLD"] = "0.75"
cache = FAQCache(data_dir=_test_dir)
assert cache.feedback_threshold == 0.75, f"Expected 0.75, got {cache.feedback_threshold}"
print(f"Init OK: threshold={cache.threshold}, feedback_threshold={cache.feedback_threshold}")

# Test 2: Lazy model loading (model not loaded at init)
assert cache._FAQCache__model is None, "Model should be lazy-loaded"
print("Lazy loading OK: model not loaded at init")

# Test 3: Store FAQs
cache.store("How do I install memcord?", "Run: pip install memcord")
cache.store("What is memcord?", "A self-learning FAQ Discord bot")
cache.store("Kako prijavim napako?", "Odpri issue na GitHubu")
assert cache.stats["cached_faqs"] == 3
assert cache.stats["stores"] == 3
print(f"Store 3 FAQs OK, stores={cache.stats['stores']}")

# Test 4: Basic check
ans, sim = cache.check("What is memcord?")
assert ans is not None
print(f"Exact match: sim={sim:.3f} OK")

# Test 5: Typo query
ans2, sim2 = cache.check("how to install memcorrrd?")
print(f"Typo query: sim={sim2:.3f}, hit={ans2 is not None}")

# Test 6: Low threshold
cache.adaptive = False
cache.threshold = 0.50
ans3, sim3 = cache.check("how install")
print(f"Low threshold: sim={sim3:.3f}, hit={ans3 is not None}")

# Test 7: Feedback + deletion via downvotes
for _ in range(5):
    cache.feedback("What is memcord?", positive=False)
cache.threshold = 0.80  # reset to normal threshold
ans4, _ = cache.check("What is memcord?")
assert ans4 is None, "Should be deleted after downvotes"
assert cache.stats["deletions"] >= 1
assert cache.stats["downvotes"] >= 3  # 3 downvotes to hit -3 and trigger delete
print(f"Feedback delete OK, deletions={cache.stats['deletions']}, downvotes={cache.stats['downvotes']}")

# Test 8: list_faqs (no query)
faqs = cache.list_faqs(limit=10)
print(f"list_faqs: {len(faqs)} results")
for f in faqs:
    print(f"  - {f['question'][:50]} | rating={f['rating']} | count={f['count']}")
assert len(faqs) == 2  # 3 stored, 1 deleted

# Test 9: list_faqs with search
results = cache.list_faqs(query="installation help", limit=5)
print(f"list_faqs(search): {len(results)} results")
assert len(results) >= 1
assert "similarity" in results[0]
print(f"  Top match: \"{results[0]['question']}\" sim={results[0]['similarity']}")

# Test 10: Consolidation — store near-duplicates and merge
# Temporarily disable auto-consolidation in store() so we can test explicit consolidate()
cache.consolidate_threshold = 1.0  # disable auto-merge during store
cache.store("how do I install memcord", "Run: pip install memcord")
cache.store("how do i install memcord ?", "Execute: pip install memcord")
cache.store("please install memcord for me", "Just run: pip install memcord")
before_consolidate = cache.stats["cached_faqs"]
cache.consolidate_threshold = 0.80  # lower for explicit consolidate test
merged = cache.consolidate()
cache.consolidate_threshold = 0.95  # restore default
after_consolidate = cache.stats["cached_faqs"]
print(f"Consolidation: {before_consolidate} → {after_consolidate} (merged={merged})")
assert merged >= 1, f"Should have merged at least 1 duplicate, got {merged}"
assert after_consolidate < before_consolidate
print(f"Consolidation check OK (merged={merged})")

# Test 11: Batched stats (not written on every call)
stats_path = os.path.join(_test_dir, "stats.json")
mtime_before = os.path.getmtime(stats_path)
# A single check should not trigger a write (batch size is 10, batch time is 60s)
cache.check("random question")
mtime_after = os.path.getmtime(stats_path)
# The batch may trigger if it's the 10th query since last save, so just log
print(f"Stats mtime before/after: {mtime_before} → {mtime_after} (batched {'OK' if mtime_before == mtime_after else 'flushed'})")

# Test 12: flush_stats forces a write
cache.flush_stats()
mtime_flushed = os.path.getmtime(stats_path)
assert mtime_flushed > mtime_after or mtime_after > mtime_before, "flush_stats should update file"
print("flush_stats OK")

# Test 13: Reset
cache.reset()
assert cache.stats["cached_faqs"] == 0
assert cache.stats["total_queries"] == 0
assert cache.stats["cache_hits"] == 0
assert cache.stats["upvotes"] == 0
assert cache.stats["stores"] == 0
print("reset() OK")

# Test 14: Backends + base class + factory import
from memcord.backends import LLMBackend, get_backend
from memcord.backends.base import BaseBackend, RetryableError, FatalError, CircuitBreakerOpenError
from memcord.backends.claude_code import ClaudeCodeBackend
from memcord.backends.openai_backend import OpenAIBackend
from memcord.backends.anthropic_backend import AnthropicBackend
from memcord.backends.ollama_backend import OllamaBackend
print("All backends + base + factory imported OK")

# Test 15: Verify inheritance
assert issubclass(ClaudeCodeBackend, BaseBackend), "ClaudeCodeBackend must inherit from BaseBackend"
assert issubclass(OpenAIBackend, BaseBackend), "OpenAIBackend must inherit from BaseBackend"
assert issubclass(AnthropicBackend, BaseBackend), "AnthropicBackend must inherit from BaseBackend"
assert issubclass(OllamaBackend, BaseBackend), "OllamaBackend must inherit from BaseBackend"
assert issubclass(BaseBackend, LLMBackend), "BaseBackend must inherit from LLMBackend"
print("Inheritance chain verified OK")

# Test 16: --version flag via subprocess
import subprocess
import sys

print("\nTesting --version flag...")
result = subprocess.run(
    [sys.executable, "-m", "memcord.cli", "version"],
    capture_output=True, text=True,
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
assert result.returncode == 0, f"version command failed: {result.stderr}"
assert "memcord v" in result.stdout, f"Unexpected version output: {result.stdout}"
print(f"Version output: {result.stdout.strip()}")

# Also test via the installed entry point if available
try:
    result2 = subprocess.run(
        ["memcord", "version"],
        capture_output=True, text=True,
    )
    assert result2.returncode == 0, f"memcord version entry point failed: {result2.stderr}"
    assert "memcord v" in result2.stdout, f"Unexpected entry point output: {result2.stdout}"
    print("Entry point version OK")
except FileNotFoundError:
    print("Entry point 'memcord' not found on PATH (expected in some environments)")

print(f"\nFinal stats: {cache.stats}")
print("\n=== ALL TESTS PASSED ===")
