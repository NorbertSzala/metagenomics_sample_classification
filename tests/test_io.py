# Norbert Optional (testy load/save TSV)


from src.utils_io import load_metadata

def test_load_metadata(tmp_path):
    tsv = tmp_path / "train.tsv"
    tsv.write_text(
        "fasta_file\tgeo_loc_name\n"
        "a.fa.gz\tCityA\n"
        "b.fa.gz\tCityB\n"
    )

    meta = load_metadata(str(tsv))

    assert len(meta) == 2
    assert meta[0].fasta_file == "a.fa.gz"
    assert meta[0].class_name == "CityA"
