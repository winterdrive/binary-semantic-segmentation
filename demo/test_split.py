"""
Test split utility for creating demo datasets.

This script randomly samples a subset of test data for demonstration purposes.
"""

import random
import os
import argparse


def create_demo_split(src_txt, dst_txt, num_samples=20):
    """
    Create a demo subset by randomly sampling from test data.
    
    Args:
        src_txt (str): Source test file path
        dst_txt (str): Destination demo file path  
        num_samples (int): Number of samples to extract
    """
    # Read all filenames
    with open(src_txt, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Random sampling
    if num_samples > len(lines):
        print(f"Sample count exceeds total test set size, using all ({len(lines)})")
        num_samples = len(lines)
    
    demo_lines = random.sample(lines, num_samples)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(dst_txt), exist_ok=True)

    # Write to demo.txt
    with open(dst_txt, "w") as f:
        for line in demo_lines:
            f.write(line + "\n")

    print(f"Randomly sampled {num_samples} samples, saved to {dst_txt}")


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description="Create demo dataset split")
    parser.add_argument("--src_txt", type=str, 
                        default="./data/oxford-iiit-pet/annotations/test.txt",
                        help="Source test file path")
    parser.add_argument("--dst_txt", type=str,
                        default="./demo/demo.txt", 
                        help="Destination demo file path")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples to extract")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Check if source file exists
    if not os.path.exists(args.src_txt):
        print(f"Error: Source file {args.src_txt} not found!")
        print("Please make sure the dataset is properly downloaded and extracted.")
        return
    
    # Create demo split
    create_demo_split(args.src_txt, args.dst_txt, args.num_samples)


if __name__ == "__main__":
    main()
