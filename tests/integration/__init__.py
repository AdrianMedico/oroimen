"""Integration tests for Hermes end-to-end flows.

Tests in this directory exercise the FULL flow:
- Message arrives (Telegram update -> aiogram Dispatcher -> handler)
- Handler processes (DB read/write, LLM call, telemetry)
- Response goes back to user

The components that are REAL:
- Database (sqlite in tmp_path)
- Telemetry (disabled, no InfluxDB)
- LLMRouter (with HTTP calls mocked via respx)
- build_message_router (with its captured closure)
- aiogram.Bot (with get_file mocked for voice tests)
- aiogram.Message (mocked - pydantic frozen, see tests/conftest.py)

The components that are MOCKED:
- httpx calls to opencode-go (via respx)
- httpx calls to Telegram for voice file download (via fake client factory)
- aiogram.Message (must mock - see rationale in tests/conftest.py)

These tests are slower than unit tests (~1-2s each) and require a network
mock setup, but they catch regression bugs that unit tests miss:
- Real interaction between DB, Router, Breaker
- Real LLM call flow with retries
- Real circuit breaker state transitions
- Real concurrency (multiple users / messages in parallel)
"""
