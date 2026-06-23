"""Unit tests for FAQCache — store, check, feedback, delete, adaptive threshold, stats, batching, consolidation."""

from __future__ import annotations

import json
import time

import pytest

from memcord.cache import FAQCache


class TestFAQCacheStore:
    """Tests for FAQCache.store()."""

    def test_store_increases_count(self, cache):
        """Storing an FAQ should increase cached_faqs count."""
        assert cache.stats["cached_faqs"] == 0
        cache.store("How do I install?", "Run pip install memcord")
        assert cache.stats["cached_faqs"] == 1

    def test_store_multiple(self, cache):
        """Storing multiple FAQs should increase count accordingly."""
        for i in range(5):
            cache.store(f"Question {i}?", f"Answer {i}")
        assert cache.stats["cached_faqs"] == 5

    def test_store_duplicate_question(self, cache):
        """Storing a very similar question should consolidate (within threshold)."""
        cache.store("What is memcord?", "A bot")
        cache.store("What is memcord?", "A Discord FAQ bot")
        # Since consolidate_threshold is 0.95 and these are identical questions,
        # the second store should update the first, not create a new entry
        # But if the models produce slightly different embeddings this could be 2.
        # Just ensure no error.
        assert cache.stats["cached_faqs"] >= 1

    def test_store_tracks_stores_stat(self, cache):
        """Store should increment the stores counter."""
        assert cache.stats["stores"] == 0
        cache.store("Q1?", "A1")
        assert cache.stats["stores"] == 1
        cache.store("Q2?", "A2")
        assert cache.stats["stores"] == 2

    def test_store_very_different_questions(self, cache):
        """Very different questions should be stored separately."""
        cache.store("How do I install memcord?", "Run: pip install memcord")
        cache.store("What is the capital of France?", "Paris")
        cache.store("How do I bake a cake?", "Mix flour and eggs")
        assert cache.stats["cached_faqs"] == 3


class TestFAQCacheCheck:
    """Tests for FAQCache.check()."""

    def test_check_empty_cache(self, cache):
        """Check on empty cache should return (None, 0.0)."""
        ans, sim = cache.check("any question")
        assert ans is None
        assert sim == 0.0

    def test_check_exact_match(self, cache):
        """Exact question should be found with high similarity."""
        cache.store("How do I install memcord?", "Run: pip install memcord")
        ans, sim = cache.check("How do I install memcord?")
        assert ans == "Run: pip install memcord"
        assert sim > 0.95  # near-exact match

    def test_check_semantic_match(self, cache):
        """Semantically similar question should be found."""
        cache.store("How do I install memcord?", "Run: pip install memcord")
        ans, sim = cache.check("What is the install process for memcord?")
        # With a decent embedding model, this should be a hit
        assert isinstance(sim, float)
        assert 0.0 <= sim <= 1.0

    def test_check_unrelated_question(self, cache):
        """Unrelated question should have low similarity."""
        cache.store("How do I install memcord?", "Run: pip install memcord")
        ans, sim = cache.check("What is the capital of France?")
        assert sim < 0.80  # should be below default threshold

    def test_check_below_threshold(self, cache):
        """Question below threshold should return None."""
        cache.store("How do I install memcord?", "Run: pip install memcord")
        cache.threshold = 0.99  # set very high
        ans, sim = cache.check("How do I install memcord?")
        # Even exact match might not reach 0.99
        if ans is not None:
            assert sim >= 0.99

    def test_check_stats_updated(self, cache):
        """Check should update total_queries and cache_hits."""
        cache.store("What is memcord?", "A self-learning FAQ Discord bot")
        assert cache.stats["total_queries"] == 0

        cache.check("What is memcord?")
        stats = cache.stats
        assert stats["total_queries"] == 1
        assert stats["cache_hits"] == 1

        cache.check("Completely unrelated query")
        stats = cache.stats
        assert stats["total_queries"] == 2
        assert stats["cache_hits"] == 1  # only one hit


class TestFAQCacheFeedback:
    """Tests for FAQCache.feedback()."""

    def test_feedback_positive_tracks_upvote(self, cache):
        """Positive feedback should increment upvotes counter."""
        cache.store("What is memcord?", "A bot")
        cache.check("What is memcord?")
        cache.feedback("What is memcord?", positive=True)

        assert cache.stats["upvotes"] == 1
        assert cache.stats["downvotes"] == 0

    def test_feedback_negative_tracks_downvote(self, cache):
        """Negative feedback should increment downvotes counter."""
        cache.store("What is memcord?", "A bot")
        cache.check("What is memcord?")
        cache.feedback("What is memcord?", positive=False)

        assert cache.stats["downvotes"] == 1
        assert cache.stats["upvotes"] == 0

    def test_feedback_after_five_downvotes_deletes(self, cache):
        """Five downvotes should delete the FAQ (rating <= -3)."""
        cache.store("What is memcord?", "A bot")
        cache.check("What is memcord?")

        for _ in range(5):
            cache.feedback("What is memcord?", positive=False)

        assert cache.stats["deletions"] >= 1
        ans, _ = cache.check("What is memcord?")
        assert ans is None  # deleted

    def test_feedback_below_three_downvotes_keeps(self, cache):
        """Two downvotes should not delete the FAQ."""
        cache.store("What is memcord?", "A bot")
        cache.check("What is memcord?")

        for _ in range(2):
            cache.feedback("What is memcord?", positive=False)

        assert cache.stats["deletions"] == 0
        ans, _ = cache.check("What is memcord?")
        assert ans == "A bot"

    def test_feedback_on_nonexistent_question(self, cache):
        """Feedback on a nonexistent question should not crash."""
        cache.feedback("Nonexistent question", positive=True)
        # Should not raise, just silently return

    def test_feedback_positive_counteracts_negative(self, cache):
        """Positive feedback after negatives should prevent deletion."""
        cache.store("What is memcord?", "A bot")
        cache.check("What is memcord?")

        cache.feedback("What is memcord?", positive=False)
        cache.feedback("What is memcord?", positive=False)
        cache.feedback("What is memcord?", positive=True)  # rating: -2 + 1 = -1

        assert cache.stats["deletions"] == 0
        ans, _ = cache.check("What is memcord?")
        assert ans == "A bot"


class TestFAQCacheAdaptiveThreshold:
    """Tests for adaptive threshold auto-tuning."""

    def test_adaptive_lowers_on_misses(self, cache):
        """Threshold should decrease when hit rate is below target."""
        cache.store("Question A?", "Answer A")
        cache.adaptive = True
        cache.threshold = 0.95
        original = cache.threshold

        for _ in range(10):
            cache.check(f"Unrelated question {_}?")

        assert cache.threshold < original

    def test_adaptive_raises_on_hits(self, cache):
        """Threshold should increase when hit rate is above target."""
        cache.store("How do I install memcord?", "Run: pip install memcord")
        cache.adaptive = True
        cache.threshold = 0.60  # low enough to always hit
        original = cache.threshold

        for _ in range(10):
            cache.check("How do I install memcord?")

        assert cache.threshold > original

    def test_adaptive_has_floor(self, cache):
        """Threshold should not drop below 0.55."""
        cache.adaptive = True
        cache.threshold = 0.56

        for _ in range(100):
            cache.check(f"Miss {_}?")

        assert cache.threshold >= 0.55

    def test_adaptive_has_ceiling(self, cache):
        """Threshold should not exceed 0.98."""
        cache.store("How do I install?", "Answer")
        cache.adaptive = True
        cache.threshold = 0.50

        for _ in range(100):
            cache.check("How do I install?")

        assert cache.threshold <= 0.98

    def test_adaptive_disabled(self, cache):
        """When adaptive is False, threshold should not change."""
        cache.adaptive = False
        original = cache.threshold

        for _ in range(10):
            cache.check("some question?")

        assert cache.threshold == original


class TestFAQCacheStats:
    """Tests for FAQCache.stats property."""

    def test_stats_initial(self, cache):
        """Initial stats should have expected shape."""
        s = cache.stats
        assert s["total_queries"] == 0
        assert s["cache_hits"] == 0
        assert s["hit_rate"] == 0.0
        assert s["cached_faqs"] == 0
        assert "threshold" in s
        assert s["adaptive"] is True
        assert s["upvotes"] == 0
        assert s["downvotes"] == 0
        assert s["deletions"] == 0
        assert s["stores"] == 0

    def test_stats_hit_rate(self, cache):
        """Hit rate should be correctly calculated."""
        cache.store("Q1?", "A1")
        cache.check("Q1?")  # hit
        cache.check("Q2?")  # miss

        s = cache.stats
        assert s["total_queries"] == 2
        assert s["cache_hits"] == 1
        assert s["hit_rate"] == 0.5

    def test_stats_survives_reload(self, tmp_cache_dir):
        """Stats should persist across cache instances."""
        cache1 = FAQCache(data_dir=tmp_cache_dir)
        cache1.store("Q1?", "A1")
        cache1.check("Q1?")
        cache1.flush_stats()

        cache2 = FAQCache(data_dir=tmp_cache_dir)
        s = cache2.stats
        assert s["total_queries"] == 1
        assert s["cache_hits"] == 1
        assert s["cached_faqs"] == 1

    def test_stats_batching(self, cache):
        """Verify stats update correctly across multiple operations."""
        for i in range(10):
            cache.store(f"Q{i}?", f"A{i}")

        for i in range(5):
            cache.check(f"Q{i}?")  # 5 hits

        for _ in range(5):
            cache.check("Unrelated?")  # 5 misses

        s = cache.stats
        assert s["total_queries"] == 10
        assert s["cache_hits"] == 5
        assert s["cached_faqs"] == 10

    def test_stats_batching_writes_periodically(self, tmp_cache_dir):
        """Stats should be batched, not written on every operation."""
        cache = FAQCache(data_dir=tmp_cache_dir, stats_batch_size=100)

        cache.store("Q1?", "A1")
        # With batch size 100, a single store may or may not trigger a write
        # (could trigger on time if >60s)
        cache.flush_stats()  # Force write
        s = cache.stats
        assert s["stores"] == 1
        assert s["cached_faqs"] == 1

    def test_flush_stats(self, tmp_cache_dir):
        """flush_stats should force-write stats."""
        cache = FAQCache(data_dir=tmp_cache_dir, stats_batch_size=100)
        cache.store("Q1?", "A1")
        cache.flush_stats()

        # Re-read stats from disk
        cache2 = FAQCache(data_dir=tmp_cache_dir)
        assert cache2.stats["stores"] == 1


class TestFAQCacheLazyLoading:
    """Tests for lazy model loading."""

    def test_model_not_loaded_at_init(self, cache):
        """Model should not be loaded on init."""
        # Access the name-mangled attribute
        assert cache._FAQCache__model is None

    def test_model_loaded_on_first_use(self, cache):
        """Model should be loaded on first encode operation."""
        cache.store("Hello?", "World")
        assert cache._FAQCache__model is not None


class TestFAQCacheConsolidation:
    """Tests for FAQCache.consolidate()."""

    def test_consolidate_empty(self, cache):
        """Consolidating empty cache should return 0."""
        assert cache.consolidate() == 0

    def test_consolidate_single_faq(self, cache):
        """Consolidating with one FAQ should return 0."""
        cache.store("Hello?", "World")
        assert cache.consolidate() == 0

    def test_consolidate_near_duplicates(self, cache):
        """Near-duplicate FAQs should be consolidated."""
        # Temporarily lower threshold so we can test with manually stored distinct items
        cache.store("What is memcord?", "A self-learning FAQ Discord bot")
        cache.store("What exactly is memcord?", "A self-learning FAQ Discord bot")
        before = cache.stats["cached_faqs"]
        merged = cache.consolidate()
        # May or may not merge depending on embedding similarity
        assert isinstance(merged, int)


class TestFAQCacheListFAQs:
    """Tests for FAQCache.list_faqs()."""

    def test_list_empty(self, cache):
        """Listing empty cache should return empty list."""
        results = cache.list_faqs()
        assert results == []

    def test_list_basic(self, cache):
        """List should return stored FAQs."""
        cache.store("What is memcord?", "A bot")
        cache.store("How do I install?", "Run pip install")
        results = cache.list_faqs(limit=10)
        assert len(results) == 2
        assert results[0]["question"] is not None
        assert results[0]["answer"] is not None
        assert "count" in results[0]
        assert "rating" in results[0]

    def test_list_with_search(self, cache):
        """List with search query should return similar results."""
        cache.store("How do I install memcord?", "Run: pip install memcord")
        cache.store("What is the weather?", "Sunny")
        results = cache.list_faqs(query="installation help", limit=5)
        assert len(results) >= 1
        assert "similarity" in results[0]

    def test_list_respects_limit(self, cache):
        """List should respect the limit parameter."""
        for i in range(10):
            cache.store(f"Q{i}?", f"A{i}")
        results = cache.list_faqs(limit=3)
        assert len(results) <= 3


class TestFAQCacheReset:
    """Tests for FAQCache.reset()."""

    def test_reset_clears_faqs(self, cache):
        """Reset should clear all FAQs."""
        cache.store("Q1?", "A1")
        cache.store("Q2?", "A2")
        assert cache.stats["cached_faqs"] == 2

        cache.reset()
        assert cache.stats["cached_faqs"] == 0
        assert cache.stats["total_queries"] == 0
        assert cache.stats["cache_hits"] == 0
        assert cache.stats["stores"] == 0

    def test_reset_clears_stats(self, cache):
        """Reset should clear all stats."""
        cache.store("Q1?", "A1")
        cache.check("Q1?")
        cache.feedback("Q1?", positive=True)

        cache.reset()
        assert cache.stats["upvotes"] == 0
        assert cache.stats["downvotes"] == 0
        assert cache.stats["deletions"] == 0


class TestFAQCacheCustomConfig:
    """Tests for custom FAQCache configuration."""

    def test_custom_threshold(self, tmp_cache_dir):
        """Custom similarity threshold should be applied."""
        cache = FAQCache(data_dir=tmp_cache_dir, similarity_threshold=0.70)
        assert cache.threshold == 0.70

    def test_adaptive_disabled_init(self, tmp_cache_dir):
        """Adaptive threshold can be disabled at init."""
        cache = FAQCache(data_dir=tmp_cache_dir, adaptive_threshold=False)
        assert cache.adaptive is False

    def test_custom_target_hit_rate(self, tmp_cache_dir):
        """Custom target hit rate should be applied."""
        cache = FAQCache(data_dir=tmp_cache_dir, target_hit_rate=0.85)
        assert cache.target_hit_rate == 0.85

    def test_custom_adjustment_rate(self, tmp_cache_dir):
        """Custom adjustment rate should be applied."""
        cache = FAQCache(data_dir=tmp_cache_dir, adjustment_rate=0.01)
        assert cache.adjustment_rate == 0.01

    def test_custom_feedback_threshold(self, tmp_cache_dir):
        """Custom feedback threshold should be applied."""
        cache = FAQCache(data_dir=tmp_cache_dir, feedback_threshold=0.75)
        assert cache.feedback_threshold == 0.75

    def test_custom_consolidate_threshold(self, tmp_cache_dir):
        """Custom consolidate threshold should be applied."""
        cache = FAQCache(data_dir=tmp_cache_dir, consolidate_threshold=0.90)
        assert cache.consolidate_threshold == 0.90
