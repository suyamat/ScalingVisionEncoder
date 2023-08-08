"""
python -m encoding.scripts.search_hparams \
    --model_name "eva02-clip-large" \
    --subject_name "all" \
    --layer_start "1" \
    --layer_step "1" \
    --layer_end "12" \
    --kernel_start "1" \
    --kernel_step "1" \
    --kernel_end "14" \
    --use_ratio "0.6"
"""

import argparse

from dotenv import load_dotenv, find_dotenv

from encoding import hparams_searching

def main(args):
    
    hparams_searching.searcher(
        args.model_name, args.subject_name, args.layer_start, args.layer_step,
        args.layer_end, args.kernel_start, args.kernel_step, args.kernel_end, args.use_ratio
    )
    

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

    parser.add_argument(
        "--layer_start",
        type=int,
        required=True,
        default=1,
        help="Number of which layer to start with.",
    )

    parser.add_argument(
        "--layer_step",
        type=int,
        required=False,
        default=1,
        help="Number of layer steps during search.",
    )
 
    parser.add_argument(   
        "--layer_end",
        type=int,
        required=True,
        help="Number of which layer to end with.",
    )

    parser.add_argument(
        "--kernel_start",
        type=int,
        required=True,
        default=1,
        help="Size of kernel to start with.",
    )

    parser.add_argument(
        "--kernel_step",
        type=int,
        required=False,
        default=1,
        help="Step size of kernel during search.",
    )
 
    parser.add_argument(   
        "--kernel_end",
        type=int,
        required=True,
        help="Size of kernel to end with.",
    )

    parser.add_argument(   
        "--use_ratio",
        type=float,
        required=True,
        default=1.0,
        help="Ratio of using sample size.",
    )

    main(parser.parse_args())