import pandas as pd
import numpy as np
import argparse
from sklearn.utils import shuffle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_orig', type=str, default='data/naive_c5_q50_s4000_v1.csv', help='Path original data')
    parser.add_argument('--path_to_samples', type=str, default='data/generated/samples_4000.csv', help='Path to generated data')
    
    args = parser.parse_args()
    return args


def main():
    # Read in original and generated data
    args = parse_args()
    orig_data = pd.read_csv(args.path_to_orig, header=None)
    generated_data = pd.read_csv(args.path_to_samples, header=None)

    # Form new data
    new_data = pd.concat([orig_data, generated_data])
    new_data = shuffle(new_data)
    
    # Write new data as CSV with appropriate title
    num_students = new_data.shape[0]
    file_path = "data/generated/naive_c5_q50_s" + str(num_students) + "_v1.csv"
    new_data.to_csv(file_path, index=False)	


if __name__ == "__main__":
    main()
