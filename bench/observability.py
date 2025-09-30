import os
import functools

try:
    from langfuse import observe as _langfuse_observe, get_client as _langfuse_get_client
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    _langfuse_observe = None
    _langfuse_get_client = None


def _check_enabled() -> bool:
    return (
        LANGFUSE_AVAILABLE 
        and os.getenv("LANGFUSE_PUBLIC_KEY") is not None
        and os.getenv("LANGFUSE_SECRET_KEY") is not None
        and os.getenv("LANGFUSE_ENABLED", "true").lower() != "false"
    )


def is_enabled() -> bool:
    """Check if Langfuse tracking is enabled."""
    return _check_enabled()


# Create observe decorator that works whether Langfuse is available or not
if LANGFUSE_AVAILABLE:
    # Langfuse is installed, use it (but still check at runtime if enabled)
    observe = _langfuse_observe
else:
    # Langfuse not installed, create no-op decorator
    def observe(*args, **kwargs):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        # Handle both @observe and @observe()
        if len(args) == 1 and callable(args[0]):
            return decorator(args[0])
        return decorator


# Create get_client function
if LANGFUSE_AVAILABLE:
    # Use real Langfuse client
    def get_client():
        """Get Langfuse client (real or dummy based on runtime config)."""
        if _check_enabled():
            return _langfuse_get_client()
        else:
            # Return dummy client even if Langfuse is installed but not configured
            return _DummyClient()
else:
    # Return dummy client if Langfuse not installed
    def get_client():
        """Get dummy Langfuse client."""
        return _DummyClient()


class _DummyClient:
    """Dummy Langfuse client that does nothing."""
    
    def update_current_trace(self, **kwargs):
        pass
    
    def update_current_span(self, **kwargs):
        pass
    
    def flush(self):
        pass
    
    def get_current_trace_id(self):
        return None


def log_status():
    """Log Langfuse status on startup."""
    enabled = _check_enabled()
    
    if enabled:
        print("Langfuse v3 tracking enabled")
    elif LANGFUSE_AVAILABLE:
        # Langfuse is installed but not all keys are present
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        
        if not public_key:
            print("Langfuse v3 installed but LANGFUSE_PUBLIC_KEY not set")
        elif not secret_key:
            print("Langfuse v3 installed but LANGFUSE_SECRET_KEY not set")
        else:
            # Keys exist but explicitly disabled
            print("Langfuse v3 installed but disabled (LANGFUSE_ENABLED=false)")
    else:
        print("Langfuse not available (tracking disabled)")


__all__ = [
    "observe",
    "get_client",
    "is_enabled",
    "log_status",
]