"""
python -m encoding.scripts.predict \
    --model_name "eva02-clip-large" \
    --subject_name "all" \
"""

import argparse

from dotenv import load_dotenv, find_dotenv

from encoding import hparams_searching

def main(args):
    
    hparams_searching.searcher(args.model_name, args.subject_name)
    

if __name__ == "__main__":
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(
        description="Search adaptive layer and kernel size of pooling by grid search."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model from which the features are extracted.",
    )

    parser.add_argument(
        "--subject_name",
        type=str,
        required=True,
        help="Name of the subject to train the model on. If you set.",
    )

    main(parser.parse_args())