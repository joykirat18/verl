#!/usr/bin/env python3
"""
Script to combine multiple parquet files into a single dataset.
Supports combining files from different directories with optional filtering and transformations.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def combine_parquet_files(
    input_files: List[str],
    output_file: str,
    add_source_column: bool = True,
    source_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> None:
    """
    Combine multiple parquet files into a single parquet file.
    
    Args:
        input_files: List of paths to input parquet files
        output_file: Path to output combined parquet file
        add_source_column: Whether to add a 'source' column indicating origin file
        source_names: Optional custom names for sources (defaults to filenames)
        verbose: Whether to print progress information
    """
    if verbose:
        print(f"Combining {len(input_files)} parquet files...")
    
    dfs = []
    
    for idx, file_path in enumerate(input_files):
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, skipping...")
            continue
            
        if verbose:
            print(f"Reading {file_path}...")
        
        df = pd.read_parquet(file_path)
        
        if add_source_column:
            if source_names and idx < len(source_names):
                source_name = source_names[idx]
            else:
                # Use parent directory name or filename
                source_name = Path(file_path).parent.name if Path(file_path).parent.name else Path(file_path).stem
            
            df['source'] = source_name
        
        if verbose:
            print(f"  Loaded {len(df)} rows from {file_path}")
        
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No valid parquet files found to combine")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    if verbose:
        print(f"\nCombined dataset statistics:")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Columns: {list(combined_df.columns)}")
        if add_source_column:
            print(f"  Source distribution:")
            print(combined_df['source'].value_counts().to_string())
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to parquet
    if verbose:
        print(f"\nSaving combined dataset to {output_file}...")
    
    combined_df.to_parquet(output_file, index=False, engine='pyarrow')
    
    if verbose:
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple parquet files into a single dataset"
    )
    parser.add_argument(
        "--input_files",
        "-i",
        nargs="+",
        required=True,
        help="List of input parquet files to combine"
    )
    parser.add_argument(
        "--output_file",
        "-o",
        required=True,
        help="Output path for combined parquet file"
    )
    parser.add_argument(
        "--source_names",
        "-s",
        nargs="+",
        default=None,
        help="Optional custom names for source column (must match number of input files)"
    )
    parser.add_argument(
        "--no_source_column",
        action="store_true",
        help="Do not add source column to track origin of each row"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate source_names if provided
    if args.source_names and len(args.source_names) != len(args.input_files):
        parser.error(
            f"Number of source names ({len(args.source_names)}) must match "
            f"number of input files ({len(args.input_files)})"
        )
    
    combine_parquet_files(
        input_files=args.input_files,
        output_file=args.output_file,
        add_source_column=not args.no_source_column,
        source_names=args.source_names,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    # Example usage for combining test datasets in this directory
    # Uncomment and modify as needed:
    
    # Option 1: Combine all test parquet files
    test_files = [
        "zebraLogic/test.parquet",
        "math-500/test.parquet",
    ]
    combine_parquet_files(
        input_files=test_files,
        output_file="combined_test_dataset/math_500_zebraLogic.parquet",
        source_names=["HuggingFaceH4/MATH-500", "WildEval/ZebraLogic"]
    )
    
    # Option 2: Use command line arguments
    # main()

