# Norbert - wczytywanie TSV, format metadata
# Utils_io

from dataclasses import dataclass
import csv

@dataclass
class metadataEntry:
    fasta_file: str
    class_name: str
    

def load_metadata(tsv_path: str)-> list[metadataEntry]:
    """
    Load tsv metadata.

    Args:
        tsv_path (str): path to tsv file

    Returns:
        list[metadataEntry] in format:
        [
            Entry(fasta_fiele="path_data1.tsv", geo_loc_name=Piastow)
        ]
    """
    
    metadata=[]
    with open(tsv_path, newline="") as f:
        # open file, each line is dict
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            entry = metadataEntry(
                fasta_file=row['fasta_file'],
                class_name=row.get("geo_loc_name")
            )
            
            metadata.append(entry)
    return metadata
