# Norbert - wczytywanie TSV, format metadata
# Utils_io

from dataclasses import dataclass
from pathlib import Path
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
    # Set correct paths for undirect paths from tsv file
    tsv_path = Path(tsv_path)
    base_dir = tsv_path.parent
    
    
    metadata=[]
    with open(tsv_path, newline="") as f:
        # open file, each line is dict
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            fasta_path = Path(row['fasta_file'])
            if not fasta_path.is_absolute():
                fasta_path = base_dir / fasta_path
                
            entry = metadataEntry(
                fasta_file=fasta_path,
                class_name=row.get("geo_loc_name")
            )
            
            metadata.append(entry)
    return metadata

def write_output_tsv(scores: dict, output_path: str):

    class_names = set(key for inner_dict in scores.values() for key in inner_dict.keys())
    sorted_classes = sorted(class_names)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')

        header = ['fasta_file'] + sorted_classes
        writer.writerow(header)

        for sample_name, scores_dict in scores.items():
            row = [sample_name]

            for class_name in sorted_classes:

                score = scores_dict[class_name]
                
                row.append(score)

            writer.writerow(row)
        
