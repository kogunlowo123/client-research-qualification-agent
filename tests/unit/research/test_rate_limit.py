from __future__ import annotations

import threading

import pytest

from client_research_agent.research.rate_limit import HostRateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)


def test_first_request_is_free_then_paced() -> None:
    clock = FakeClock()
    limiter = HostRateLimiter(2.0, clock=clock, sleep=clock.sleep)
    assert limiter.acquire("www.sec.gov") == 0.0
    assert limiter.acquire("www.sec.gov") == pytest.approx(0.5)
    assert limiter.acquire("www.sec.gov") == pytest.approx(1.0)
    assert clock.sleeps == [pytest.approx(0.5), pytest.approx(1.0)]


def test_tokens_refill_with_time() -> None:
    clock = FakeClock()
    limiter = HostRateLimiter(1.0, clock=clock, sleep=clock.sleep)
    limiter.acquire("a.example")
    clock.now += 1.0
    assert limiter.acquire("a.example") == 0.0
    clock.now += 0.25
    assert limiter.acquire("a.example") == pytest.approx(0.75)


def test_hosts_are_independent_and_case_insensitive() -> None:
    clock = FakeClock()
    limiter = HostRateLimiter(1.0, clock=clock, sleep=clock.sleep)
    assert limiter.acquire("A.example") == 0.0
    assert limiter.acquire("b.example") == 0.0
    assert limiter.acquire("a.example.") == pytest.approx(1.0)


def test_burst_allows_initial_parallel_requests() -> None:
    clock = FakeClock()
    limiter = HostRateLimiter(1.0, burst=3.0, clock=clock, sleep=clock.sleep)
    waits = [limiter.acquire("h.example") for _ in range(4)]
    assert waits == [0.0, 0.0, 0.0, pytest.approx(1.0)]


def test_crawl_delay_only_slows_a_host_down() -> None:
    clock = FakeClock()
    limiter = HostRateLimiter(2.0, burst=2.0, clock=clock, sleep=clock.sleep)
    limiter.set_min_interval("slow.example", 5.0)
    limiter.set_min_interval("slow.example", 0.1)
    limiter.set_min_interval("slow.example", 0)
    assert limiter.rate_for("slow.example") == pytest.approx(0.2)
    assert limiter.rate_for("other.example") == pytest.approx(2.0)
    assert limiter.acquire("slow.example") == 0.0
    assert limiter.acquire("slow.example") == pytest.approx(5.0)


def test_concurrent_reservations_are_serialised() -> None:
    clock = FakeClock()
    limiter = HostRateLimiter(10.0, clock=clock, sleep=clock.sleep)
    waits: list[float] = []
    lock = threading.Lock()

    def worker() -> None:
        wait = limiter.acquire("h.example")
        with lock:
            waits.append(wait)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(round(w, 6) for w in waits) == [round(i * 0.1, 6) for i in range(20)]


@pytest.mark.parametrize(("rate", "burst"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.5)])
def test_rejects_invalid_configuration(rate: float, burst: float) -> None:
    with pytest.raises(ValueError, match="must be"):
        HostRateLimiter(rate, burst=burst)
