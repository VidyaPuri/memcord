"""Quick import verification for MemcordBot — no heavy inspect."""
print("Importing...")
from memcord.discord_ import MemcordBot
from memcord.discord_.bot import _sanitize_input, _chunk_text, _get_bot, faq_list
print("Bot import OK")

# Check class attributes
assert hasattr(MemcordBot, "is_memcord"), "Missing is_memcord flag"
assert MemcordBot.is_memcord is True
print("✓ is_memcord = True")

# Check key methods
assert hasattr(MemcordBot, "_check_rate_limit"), "Missing rate limiting"
print("✓ _check_rate_limit")

assert hasattr(MemcordBot, "_send_chunked"), "Missing response chunking"  
print("✓ _send_chunked")

assert hasattr(MemcordBot, "_graceful_shutdown"), "Missing graceful shutdown"
print("✓ _graceful_shutdown")

assert hasattr(MemcordBot, "_handle_shutdown_signal"), "Missing signal handler"
print("✓ _handle_shutdown_signal")

assert hasattr(MemcordBot, "_setup_signal_handlers"), "Missing signal setup"
print("✓ _setup_signal_handlers")

# Check instance attrs are created in __init__ (check source)
import inspect
init_src = inspect.getsource(MemcordBot.__init__)
assert "self._rate_limits" in init_src, "Missing rate limit storage in __init__"
print("✓ _rate_limits created in __init__")

assert "self._context" in init_src, "Missing context storage in __init__"
print("✓ _context created in __init__")

# Check helpers
print("✓ _sanitize_input imported")
print("✓ _chunk_text imported")  
print("✓ _get_bot imported")
print("✓ faq_list command imported")

# Verify _sanitize_input works
test = _sanitize_input("<@123> hello <@456> world")
assert test == "hello  world", f"Expected 'hello  world', got '{test}'"
print("✓ _sanitize_input strips mentions")

test2 = _sanitize_input("   ")
assert test2 == "", f"Expected '', got '{test2}'"
print("✓ _sanitize_input handles whitespace-only")

# Verify _chunk_text works
assert _chunk_text("hello") == ["hello"]
print("✓ _chunk_text handles short text")

long = "x" * 2500
chunks = _chunk_text(long)
assert len(chunks) > 1
assert all(len(c) <= 1900 for c in chunks)
print(f"✓ _chunk_text splits long text into {len(chunks)} chunks")

print("\n=== All hardening features verified ===")
