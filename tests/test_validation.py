
from src.validation import measure_time
import time


def slow_function(x):
    time.sleep(0.1)
    return x * 2


def test_measure_time_basic():
    stats = measure_time(slow_function, 3)

    assert "result" in stats
    assert "total_time" in stats
    assert stats["result"] == 6
    assert stats["total_time"] >= 0.1
