# norbert (optional) test sketches

# tests/test_sketch.py

import gzip
from src.compute_sketch import compute_sketch

def write_fasta_gz(path, seqs):
    with gzip.open(path, "wt") as f:
        for i, s in enumerate(seqs):
            f.write(f">seq{i}\n{s}\n")

def test_basic_sketch(tmp_path):
    fasta = tmp_path / "test.fa.gz"
    write_fasta_gz(fasta, ["ACGTACGTACGT"])

    sketch = compute_sketch(str(fasta), k=3, sketch_size=5)

    assert isinstance(sketch, list)
    assert len(sketch) <= 5
    assert all(isinstance(h, int) for h in sketch)

def test_reproducibility(tmp_path):
    fasta = tmp_path / "test.fa.gz"
    write_fasta_gz(fasta, ["ACGTACGTACGT"])

    s1 = compute_sketch(str(fasta), k=4, sketch_size=10)
    s2 = compute_sketch(str(fasta), k=4, sketch_size=10)

    assert s1 == s2  # deterministic hashing

def test_sorted_output(tmp_path):
    fasta = tmp_path / "test.fa.gz"
    write_fasta_gz(fasta, ["ATGCTAGCTAGC"])

    sketch = compute_sketch(str(fasta), k=3, sketch_size=3)

    assert sketch == sorted(sketch)

def test_multiple_sequences(tmp_path):
    fasta = tmp_path / "test.fa.gz"
    write_fasta_gz(fasta, ["AAAAA", "TTTTT"])

    sketch = compute_sketch(str(fasta), k=2, sketch_size=4)

    assert len(sketch) <= 4
    

