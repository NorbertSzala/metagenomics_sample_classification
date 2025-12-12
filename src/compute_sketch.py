# Norbert - implementacja sketchów
# compute_sketch.py
import gzip
import mmh3
import heapq

def compute_sketch(fasta_path, k=21, sketch_size=2000):
    """
    Compute a MinHash sketch from a gzipped FASTA file.

    Parameters:
        fasta_path (str): path to gzipped FASTA file (.gz)
        k (int): k-mer length
        sketch_size (int): number of minimum hashes to keep

    Time complexity:
        s = sketch size, k = kmer length, N total number of bases across all reads
        Hashing each k-mer:   O(1)        (string slice + mmh3)
        Heap update:          O(log s)
        Total: O(N log s), where s ~2000 -> effectively O(N)

    Space complexity:
        Only one sequence is keeped in RAM -> O(L)
        Max heap of size s -> O(s)
        O(L + s)
    Returns:
        list[int]: sorted list of sketch_size smallest hash values
    """
    
    # in python there is only min_heap, thats we will inverse numbers, 1 -> -1
    max_heap = []
    
    # ----- Fasta .gz reader -----
    with gzip.open(fasta_path, 'rt') as f:
        seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue

        
            if line.startswith(">"):
                # process if previous sequence is in 'seq' variable
                if seq:
                    process_sequence("".join(seq), k, sketch_size, max_heap)
                    seq = []
                    
            else:
                seq.append(line)
        
        # process last sequence
        if seq:
            process_sequence("".join(seq), k, sketch_size, max_heap)
    
    # convert MAX-HEAP to normal values (positive)
    sketch = sorted([-h for h in max_heap])
    
    return sketch
    


def process_sequence(seq:str, k:int, sketch_size:int, max_heap:list):
    """
    Extract k-mers from one sequence and update the global max_heap

    We didnt use rolling hash (Rabin hash for example) becouse it can generate 
    colisions what can affect on Jaccard AUC worse results what is not compensed 
    by better complexity
    
    Args:
        seq (str): your sequence iterated in compute_sketch
        k (int): kmer size
        sketch_size (int): sketch size
        max_heap (list): updated heap
    """
    
    n = len(seq)
    if n<k:
        return
    
    # fast generation k-mers. Slicing + iterations 
    for i in range(n-k+1):
        kmer = seq[i:i+k]
        h, _ = mmh3.hash64(kmer, signed = False)
        
        # if heap is nto full, add last hash (negative hash, because its max heap)
        if len(max_heap) < sketch_size:
            heapq.heappush(max_heap, -h)
        else:
            # if heap is full, add only when new hash is smaller than the biggest hash
            if h < -max_heap[0]:
                heapq.heapreplace(max_heap, -h)