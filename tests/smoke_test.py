"""Smoke test for memcord — cache + backends."""
import shutil
shutil.rmtree("/tmp/test_memcord", ignore_errors=True)

from memcord.cache import FAQCache

cache = FAQCache(data_dir="/tmp/test_memcord")
print(f"Init OK: threshold={cache.threshold}")

cache.store("How do I install memcord?", "Run: pip install memcord")
cache.store("What is memcord?", "A self-learning FAQ Discord bot")
cache.store("Kako prijavim napako?", "Odpri issue na GitHubu")
assert cache.stats["cached_faqs"] == 3
print("Store 3 FAQs OK")

ans, sim = cache.check("What is memcord?")
assert ans is not None
print(f"Exact match: sim={sim:.3f} OK")

ans2, sim2 = cache.check("how to install memcorrrd?")
print(f"Typo query: sim={sim2:.3f}, hit={ans2 is not None}")

cache.adaptive = False
cache.threshold = 0.50
ans3, sim3 = cache.check("how install")
print(f"Low threshold: sim={sim3:.3f}, hit={ans3 is not None}")

for _ in range(5):
    cache.feedback("What is memcord?", positive=False)
ans4, _ = cache.check("What is memcord?")
assert ans4 is None, "Should be deleted after downvotes"
print("Feedback delete OK")

from memcord.backends.claude_code import ClaudeCodeBackend
from memcord.backends.openai_backend import OpenAIBackend
from memcord.backends.anthropic_backend import AnthropicBackend
from memcord.backends.ollama_backend import OllamaBackend
print("All backends imported OK")

print(f"\nFinal stats: {cache.stats}")
print("\n=== ALL TESTS PASSED ===")
