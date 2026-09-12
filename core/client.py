#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core.client - TimechoAI Client Factory

Purpose: Unified management of TimechoAIClient / TimechoAIAsyncClient instance
         creation, shielding API_KEY retrieval logic.

Design Goals: All business modules (core, testcases) should obtain clients through this factory, in order to:
    1. API_KEY is only read from config.settings, one modification takes effect globally
    2. Mock clients can be injected here in the future, facilitating testing
    3. Only need to modify this file when SDK constructor signature changes

Calling Convention:
    Recommended way:
        from core.timecho import forecast    # Business modules use indirectly through core
    Avoided way:
        from core.client import get_timecho_client # Business modules should not call directly
        from timecho_ai import TimechoAIClient # Business modules should not reference SDK directly

  In the entire project, only core.timecho is the direct caller of core.client.
  All test scripts under testcases/ use this factory indirectly through core.timecho.forecast().

Author: Janesong
Create Date: 2026/06/29.
"""

import asyncio
from timecho_ai import TimechoAIClient, TimechoAIAsyncClient
from config.settings import API_KEY

# ---------------------------------------------------------------------------
# Sync client singleton
# ---------------------------------------------------------------------------
_client: TimechoAIClient | None = None

# ---------------------------------------------------------------------------
# Async client singleton
# ---------------------------------------------------------------------------
_async_client: TimechoAIAsyncClient | None = None


# ===========================================================================
#  Sync client
# ===========================================================================

def get_timecho_client(api_key: str | None = None) -> TimechoAIClient:
    """
    Get the TimechoAIClient singleton instance.

    Automatically warms up the connection upon the first call to prevent
    potential failures during the initial API request.

    Args:
        api_key: The API key string. If None (default), it is automatically
                 read from `config.settings.API_KEY`.
                 Priority: Explicit argument > Environment variable `TIMECHO_API_KEY` > Default value.

    Returns:
        The TimechoAIClient instance.
    """
    global _client

    if _client is None:
        _client = TimechoAIClient(api_key=api_key or API_KEY)

        # Warm-up: triggers underlying initialization to prevent DNS resolution failure on the first call.
        _warmup_client(_client)

    return _client


def _warmup_client(client: TimechoAIClient) -> None:
    """
    Warm up sync client connection.

    Args:
        client: TimechoAIClient instance.
    """
    try:
        # Directly triggers Session creation without making any API calls.
        client._get_session()
    except Exception as exp:
        # Warm-up failure does not block the execution flow.
        print(f"[Warning] SDK warm-up failed (ignorable): {type(exp).__name__}")
        # Do not raise an exception; allows for subsequent retries.


def reset_client() -> None:
    """
    Reset the sync client instance.

    Used in testing scenarios to force the client to be recreated on the next call.
    """
    global _client
    if _client is not None:
        try:
            _client.close()
        except:
            pass
    _client = None


# ===========================================================================
#  Async client
# ===========================================================================

def get_timecho_async_client(api_key: str | None = None) -> TimechoAIAsyncClient:
    """
    Get the TimechoAIAsyncClient singleton instance.

    The async client shares the same API_KEY resolution logic as the sync
    client.  The first call triggers an asynchronous warm-up so that the
    underlying aiohttp / httpx session is ready before real requests.

    Args:
        api_key: The API key string. If None (default), it is automatically
                 read from `config.settings.API_KEY`.
                 Priority: Explicit argument > Environment variable `TIMECHO_API_KEY` > Default value.

    Returns:
        The TimechoAIAsyncClient instance.
    """
    global _async_client

    if _async_client is None:
        _async_client = TimechoAIAsyncClient(api_key=api_key or API_KEY)

        # Warm-up: triggers underlying async session creation.
        _warmup_async_client(_async_client)

    return _async_client


def _warmup_async_client(client: TimechoAIAsyncClient) -> None:
    """
    Warm up async client connection.

    Tries the async warm-up path first; if no event-loop is running (e.g.
    the caller is in a plain synchronous context), falls back to creating
    a temporary loop so that session initialization still happens eagerly.

    Args:
        client: TimechoAIAsyncClient instance.
    """

    async def _do_warmup() -> None:
        # Trigger async session creation without making any API calls.
        await client._get_async_session()

    try:
        # Case 1: an event loop is already running — schedule the coroutine.
        loop = asyncio.get_running_loop()
        loop.create_task(_do_warmup())
    except RuntimeError:
        # Case 2: no running loop — create one just for warm-up.
        try:
            asyncio.run(_do_warmup())
        except Exception as exp:
            print(f"[Warning] Async SDK warm-up failed (ignorable): {type(exp).__name__}")
    except Exception as exp:
        print(f"[Warning] Async SDK warm-up failed (ignorable): {type(exp).__name__}")


def reset_async_client() -> None:
    """
    Reset the async client instance.

    Properly closes the underlying async session before discarding the
    singleton.  Safe to call from both sync and async contexts.
    """
    global _async_client
    if _async_client is not None:
        try:
            async def _close():
                await _async_client.aclose()

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_close())
            except RuntimeError:
                asyncio.run(_close())
        except Exception:
            pass
    _async_client = None
