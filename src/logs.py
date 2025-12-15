from pathlib import Path
import time
import os
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None


class RunLogger:
    def __init__(self, n_units: int | None = None, log_dir: str = "logs"):
        """
        Initialize the logger
        
        n_units: number of units/probes (eg. number of FASTA files etc)
            Used to compute average time per unit
        
        log_dir = firectory where log files will be saveds 
        """
        self.n_units = n_units # Number of processed units
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.start_time = None
        self.end_time = None

        self.process = psutil.Process(os.getpid()) if psutil else None # Current process handle
        self.start_ram = None # RAM usage at the start of the run (MB)
        self.peak_ram = 0.0  # Maximum RAM usage observed during the run (MB)

    def __enter__(self):
        """
        Starts automatically to measure and record initial RAM usage
        """
        self.start_time = time.perf_counter()
        if self.process:
            self.start_ram = self.process.memory_info().rss / 1024**2 # Measure initial RAM usage
        return self

    def update_peak_ram(self):
        """
        Update the highest RAM usage value.

        Is called at steps which require high RAM usage (e.g. after sketching, model building, classification).
        """
        if self.process:
            current = self.process.memory_info().rss / 1024**2
            self.peak_ram = max(self.peak_ram, current)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Starts automatically to stop time measurements and safe log files
        """
        self.end_time = time.perf_counter()
        self._write_log()
        
    def _write_log(self):
        """
        Write timing and RAM usage statistics to a log file.
        """
        total_time = self.end_time - self.start_time
        avg_time = (
            total_time / self.n_units
            if self.n_units
            else None
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"run_{timestamp}.log"

        with open(log_file, "w") as f:
            f.write("=== RUN LOG ===\n")
            f.write(f"Total time [s]: {total_time:.4f}\n")

            if avg_time is not None:
                f.write(f"Avg time per unit [s]: {avg_time:.6f}\n")

            if self.process:
                f.write(f"Start RAM [MB]: {self.start_ram:.2f}\n")
                f.write(f"Peak RAM [MB]: {self.peak_ram:.2f}\n")
            else:
                f.write("RAM measurement: psutil not available\n")

    def get_stats(self):
        """Return total time and peak RAM."""
        total_time = self.end_time - self.start_time
        return {
            "total_time": total_time,
            "peak_ram": self.peak_ram
        }