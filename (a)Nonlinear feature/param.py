import argparse


def parameter_parser():

    parser = argparse.ArgumentParser(description="Run LDA-GMCB.")

    parser.add_argument("--dataset-path",
                        nargs="?",
                        default="../data2",
                        help="Training datasets.")

    parser.add_argument("--epoch",
                        type=int,
                        default=400,
                        help="Number of training epochs. Default is 400.")

    parser.add_argument("--gcn-layers",
                        type=int,
                        default=2,
                        help="Number of Graph Convolutional Layers. Default is 2.")

    parser.add_argument("--out-channels",
                        type=int,
                        default=128,
                        help="out-channels of cnn. Default is 128.")

    parser.add_argument("--lncRNA-number",
                        type=int,
                        default=89,
                        help="lncRNA number. Default is 89.")

    parser.add_argument("--fcir",
                        type=int,
                        default=64,
                        help="lncRNA feature dimensions. Default is 64.")

    parser.add_argument("--disease-number",
                        type=int,
                        default=190,
                        help="disease number. Default is 190.")

    parser.add_argument("--fdis",
                        type=int,
                        default=64,
                        help="disease feature dimensions. Default is 64.")


    return parser.parse_args()