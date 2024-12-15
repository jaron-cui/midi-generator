# Programmatic Music Generator

## Quickstart
This project uses Python 3.11. Using an older version may result in an error during dependency installation
having to do with Cargo, the package manager for Rust.

1. Run `pip install -r requirements.txt`.
2. Go to the `src/synthetic_music_generation/` directory. e.g. `cd src/synthetic_music_generation`
3. Run `python generate.py`, or run `src/synthetic_music_generation/generate.py` from your IDE of choice.

This program will generate and play a unique piece for you.
Make sure your audio is enabled. Try running it multiple times
to see the different kinds of results you may hear.

## Datasets

If you want to generate a synthetic pieces in bulk, such as for a dataset, refer to the
`generate_dataset(<folder>, <num_files>)` function defined in `src/synthetic_music_generation/dataset.py`.

There are additional rough functions and code excerpts in `src/data_processing/create_dict.py`
for generating many 'epochs' worth of synthetic data, preprocessing it, and pickling it into archive files that are
faster to load.

## Other

Check out `output/synthetic_data/interesting_synthetic_examples` for a few cool example pieces
generated using this project.