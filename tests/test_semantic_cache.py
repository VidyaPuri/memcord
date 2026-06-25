"""Unit tests for SemanticAnswerCache — scope, should_cache, validate, promote_after, build_cache."""

from __future__ import annotations

from memcord.cache import CacheHit, SemanticAnswerCache, build_cache


# ── scope isolation ────────────────────────────────────────


class TestScopeIsolation:
    """Scope is an opaque partition key — entries under one scope are invisible to others."""

    def test_same_scope_hit(self, semantic_cache):
        """Entry written under scope='a' is found when queried with scope='a'."""
        semantic_cache.observe("hello?", "world A", scope="a")
        hit = semantic_cache.lookup("hello?", scope="a")
        assert hit is not None
        assert hit.answer == "world A"

    def test_different_scope_miss(self, semantic_cache):
        """Entry written under scope='a' is NOT found when queried with scope='b'."""
        semantic_cache.observe("hello?", "world A", scope="a")
        hit = semantic_cache.lookup("hello?", scope="b")
        assert hit is None

    def test_scopes_are_independent(self, semantic_cache):
        """Two entries with same question but different scopes coexist."""
        semantic_cache.observe("question?", "answer-alpha", scope="alpha")
        semantic_cache.observe("question?", "answer-beta", scope="beta")

        hit_a = semantic_cache.lookup("question?", scope="alpha")
        hit_b = semantic_cache.lookup("question?", scope="beta")

        assert hit_a is not None and hit_a.answer == "answer-alpha"
        assert hit_b is not None and hit_b.answer == "answer-beta"

    def test_empty_scope_not_leaking(self, semantic_cache):
        """Scope 'c' that never had entries returns None."""
        semantic_cache.observe("greeting?", "hi", scope="a")
        hit = semantic_cache.lookup("greeting?", scope="c")
        assert hit is None


# ── promote_after ──────────────────────────────────────────


class TestPromoteAfter:
    """Entries are invisible to lookup until observed promote_after times."""

    def test_promote_after_1_immediate(self, semantic_cache):
        """promote_after=1 means the entry is immediately retrievable."""
        cache = SemanticAnswerCache(data_dir=semantic_cache._stats_path.parent, promote_after=1)
        cache.observe("q?", "answer")
        hit = cache.lookup("q?")
        assert hit is not None
        assert hit.answer == "answer"

    def test_promote_after_3_needs_three_observations(self, semantic_cache):
        """Entry must be observed 3 times before lookup returns it."""
        d = str(semantic_cache._stats_path.parent) + "_pa3"
        cache = SemanticAnswerCache(data_dir=d, promote_after=3)

        cache.observe("question?", "answer", scope="test")
        assert cache.lookup("question?", scope="test") is None  # 1/3

        cache.observe("question?", "answer", scope="test")
        assert cache.lookup("question?", scope="test") is None  # 2/3

        cache.observe("question?", "answer", scope="test")
        hit = cache.lookup("question?", scope="test")  # 3/3
        assert hit is not None
        assert hit.answer == "answer"

    def test_promote_after_per_scope(self, semantic_cache):
        """promote_after is global, but observation counts are per-scope."""
        d = str(semantic_cache._stats_path.parent) + "_pa_scope"
        cache = SemanticAnswerCache(data_dir=d, promote_after=2)

        # Observe in scope 'a' twice
        cache.observe("q?", "answer A", scope="a")
        cache.observe("q?", "answer A", scope="a")
        assert cache.lookup("q?", scope="a") is not None  # promoted in 'a'

        # Observe in scope 'b' only once
        cache.observe("q?", "answer B", scope="b")
        assert cache.lookup("q?", scope="b") is None  # not yet promoted in 'b'


# ── should_cache hook ──────────────────────────────────────


class TestShouldCache:
    """should_cache(answer) gates eligibility — False drops the answer."""

    def test_should_cache_false_drops(self, semantic_cache):
        """Answers for which should_cache returns False are never stored."""
        d = str(semantic_cache._stats_path.parent) + "_sc"
        cache = SemanticAnswerCache(
            data_dir=d, should_cache=lambda a: "good" in a
        )
        cache.observe("q?", "bad answer", scope="test")
        cache.observe("q?", "good answer", scope="test")

        assert cache.stats["stores"] == 1
        hit = cache.lookup("q?", scope="test")
        assert hit is not None
        assert hit.answer == "good answer"

    def test_should_cache_default_caches_everything(self, semantic_cache):
        """Default should_cache (not set) caches all answers."""
        cache = SemanticAnswerCache(data_dir=str(semantic_cache._stats_path.parent) + "_scd")
        cache.observe("q1?", "any answer")
        cache.observe("q2?", "")
        assert cache.stats["stores"] == 2

    def test_should_cache_empty_answer(self, semantic_cache):
        """should_cache can filter out empty answers."""
        d = str(semantic_cache._stats_path.parent) + "_sce"
        cache = SemanticAnswerCache(
            data_dir=d, should_cache=lambda a: len(a) > 0
        )
        cache.observe("q?", "", scope="test")
        assert cache.stats["stores"] == 0


# ── validate hook ──────────────────────────────────────────


class TestValidate:
    """validate(hit) filters lookup candidates — False discards the hit."""

    def test_validate_false_causes_miss(self, semantic_cache):
        """When validate returns False, lookup returns None."""
        d = str(semantic_cache._stats_path.parent) + "_val"
        cache = SemanticAnswerCache(
            data_dir=d, validate=lambda h: "approved" in h.answer
        )
        cache.observe("q?", "rejected answer", scope="test")
        hit = cache.lookup("q?", scope="test")
        assert hit is None

    def test_validate_true_allows_hit(self, semantic_cache):
        """When validate returns True, lookup succeeds."""
        d = str(semantic_cache._stats_path.parent) + "_val2"
        cache = SemanticAnswerCache(
            data_dir=d, validate=lambda h: "approved" in h.answer
        )
        cache.observe("q?", "approved answer", scope="test")
        hit = cache.lookup("q?", scope="test")
        assert hit is not None
        assert hit.answer == "approved answer"

    def test_validate_receives_cache_hit(self, semantic_cache):
        """validate receives a CacheHit with the correct id and answer."""
        d = str(semantic_cache._stats_path.parent) + "_val3"
        received: list[CacheHit] = []

        def record(hit: CacheHit) -> bool:
            received.append(hit)
            return True

        cache = SemanticAnswerCache(data_dir=d, validate=record)
        cache.observe("q?", "answer", scope="test")
        cache.lookup("q?", scope="test")

        assert len(received) == 1
        assert received[0].id != ""
        assert received[0].answer == "answer"


# ── vote method ────────────────────────────────────────────


class TestVote:
    """vote(hit_id, delta) adjusts rating and prunes on low score."""

    def test_vote_positive(self, semantic_cache):
        """Positive vote increments upvotes counter."""
        semantic_cache.observe("q?", "answer", scope="test")
        hit = semantic_cache.lookup("q?", scope="test")
        assert hit is not None

        before = semantic_cache.stats["upvotes"]
        semantic_cache.vote(hit.id, 1)
        assert semantic_cache.stats["upvotes"] == before + 1

    def test_vote_negative(self, semantic_cache):
        """Negative vote increments downvotes counter."""
        semantic_cache.observe("q?", "answer", scope="test")
        hit = semantic_cache.lookup("q?", scope="test")
        assert hit is not None

        before = semantic_cache.stats["downvotes"]
        semantic_cache.vote(hit.id, -1)
        assert semantic_cache.stats["downvotes"] == before + 1

    def test_vote_nonexistent_id(self, semantic_cache):
        """Vote on a nonexistent ID is a no-op — does not crash."""
        semantic_cache.vote("nonexistent-id-12345", 1)
        # Should not raise

    def test_vote_downvote_prune(self, semantic_cache):
        """Enough downvotes (rating ≤ -3) prune the entry."""
        semantic_cache.observe("q?", "answer", scope="test")
        hit = semantic_cache.lookup("q?", scope="test")
        assert hit is not None

        for _ in range(4):
            semantic_cache.vote(hit.id, -1)

        assert semantic_cache.stats["deletions"] >= 1
        # Entry should be gone
        result = semantic_cache.lookup("q?", scope="test")
        assert result is None


# ── build_cache factory ────────────────────────────────────


class TestBuildCache:
    """build_cache() factory creates a working SemanticAnswerCache."""

    def test_build_cache_creates_cache(self, tmp_cache_dir):
        """Factory returns a SemanticAnswerCache that works."""
        cache = build_cache(data_dir=tmp_cache_dir)
        assert isinstance(cache, SemanticAnswerCache)

        cache.observe("hello?", "world", scope="test")
        hit = cache.lookup("hello?", scope="test")
        assert hit is not None
        assert hit.answer == "world"

    def test_build_cache_passes_options(self, tmp_cache_dir):
        """Provider options are forwarded to SemanticAnswerCache."""
        cache = build_cache(
            data_dir=tmp_cache_dir,
            similarity_threshold=0.70,
            adaptive_threshold=False,
        )
        assert cache.threshold == 0.70
        assert cache.adaptive is False

    def test_build_cache_promote_after(self, tmp_cache_dir):
        """promote_after is applied."""
        cache = build_cache(data_dir=tmp_cache_dir, promote_after=2)
        cache.observe("q?", "answer")
        assert cache.lookup("q?") is None
        cache.observe("q?", "answer")
        assert cache.lookup("q?") is not None


# ── smoke: all features together ───────────────────────────


class TestIntegration:
    """End-to-end integration of all features."""

    def test_full_pipeline(self, tmp_cache_dir):
        """scope + promote_after + should_cache + validate all work together."""
        cache = build_cache(
            data_dir=tmp_cache_dir,
            promote_after=2,
            should_cache=lambda a: len(a) > 5,
            validate=lambda h: "safe" in h.answer.lower(),
            scope_default="main",
        )

        # Short answer — should_cache rejects
        cache.observe("q1?", "bad")
        assert cache.stats["stores"] == 0

        # Long answer but wrong scope test for validate
        cache.observe("q1?", "a long but unsafe response", scope="main")
        assert cache.stats["stores"] == 1

        # Not yet promoted (only 1 observation)
        hit = cache.lookup("q1?", scope="main")
        assert hit is None  # not promoted + would fail validate anyway

        # Observe again — now promoted
        cache.observe("q1?", "a long safe response", scope="main")
        hit = cache.lookup("q1?", scope="main")
        assert hit is not None
        assert "safe" in hit.answer.lower()
