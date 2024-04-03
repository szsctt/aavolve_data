#!/usr/bin/env python3

# script to filter consesus reads by number of repeats

import argparse
import gzip
import pdb

def main():
    
    # command line arguements
    parser  = argparse.ArgumentParser(description="Filter consensus reads by number of repeats")
    parser.add_argument("-i", "--input", help="input file", required=True)
    parser.add_argument("-o", "--output", help="output directory", required=True)
    args= parser.parse_args()
        
    highest = 10
    # generate output filenames for each nubmer of repeats
    output_files = {i: open(args.output + f"/{i}_repeats.fa", "w") for i in range(0, highest + 1)}

    # open input file
    opener = gzip.open if args.input.endswith(".gz") else open
    with opener(args.input, "rt") as infile:
        # read file two lines at a time
        for line in infile:
            if line[0] == ">":
                # get number of repeats
                repeats = int(line.split("_")[3])
                
                # write to file  if more than highest nubmer of repeats
                if repeats > highest:
                    repeats = highest

                # write to appropriate file
                output_files[repeats].write(line)
            else:
                output_files[repeats].write(line)
        
    # close output files
    for i in output_files.keys():
        output_files[i].close()


if __name__ == "__main__":
    main()
