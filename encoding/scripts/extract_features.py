import argparse

from dotenv import find_dotenv, load_dotenv

from encoding import extracting

"""
python -m encoding.scripts.extract_features \
    --model_name "InternImage" \
    --subject_name "all" \
    --skip "2" \
    --n_device "4" \
    --batch_size "16"

"""

def main(args):
    """Main entrypoint of the script."""

    extracting.inference(
        args.model_name,
        args.subject_name,
        args.skip,
        args.n_device,
        args.batch_size,
    )
    

if __name__ == "__main__":
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(
        description="Train a model to predict brain activity from music embeddings."
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

    parser.add_argument(
        "--skip",
        type=int,
        required=True,
        help="Number of layer skip.",
    )

    parser.add_argument(
        "--n_device",
        type=int,
        required=True, 
        help="Number of gpu device.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=True, 
        help="Batch size for inference.",
    )

    main(parser.parse_args())