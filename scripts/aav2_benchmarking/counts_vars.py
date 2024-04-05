#!/usr/bin/env python

# count variants in a tsv file, and output per-base and per-read substitution, deletion and insertion counts, as well as overall accuracy

import argparse

from aavolve.pivot_variants_to_wide import get_reads 
from aavolve.utils import get_repeats_from_r2c2_name

def main():

    # get command line args
    parser = argparse.ArgumentParser(description="Caculate accuracy from a tsv file")
    parser.add_argument("--variants", '-v', help="Input variant tsv file", required=True)
    parser.add_argument("--read-ids", '-r', help="Input read ids file", required=True)
    parser.add_argument("--reference-length", "-l", help="Reference length", required=True, type=int)
    parser.add_argument("--output", "-o", help="Output file", default="output.tsv")
    args = parser.parse_args()

    counts = {} # track total number of ins, del, sub, reads
    per_read = {} # track per-read ins, del, sub, acc
    for rid, vars in get_reads(args.variants, args.read_ids):
        
        #analyse each number of repeats separately
        repeats = get_repeats_from_r2c2_name(rid)
        if repeats not in counts.keys():
            counts[repeats] = {'ins':0, 'del': 0, 'sub': 0, 'reads':0}

        # increment read count
        counts[repeats]['reads'] += 1

        # count number of insertions, deletions and substitutions in read    
        for var in vars.values():
            # don't count variants that are beyond the reference length
            if isinstance(var.zero_pos(), str):
                pos = int(var.zero_pos().split("_")[0])
            else:
                pos = var.zero_pos()
            if pos >= args.reference_length:
                continue
            # increment count
            counts[repeats][var.var_type] += 1



    calc = {}

    # calculate per-read and per-base error rates
    for repeats, count in counts.items():

        calc[repeats] = {
            'repeats': repeats,
            'reads': count['reads'],
            'ins': count['ins'],
            'del': count['del'],
            'sub': count['sub'],
            'total_vars': count['ins']+count['del']+count['sub'],
            'ins_per_read': count['ins']/count['reads'],
            'del_per_read': count['del']/count['reads'],
            'sub_per_read': count['sub']/count['reads'],
            'total_variants_per_read': (count['ins']+count['del']+count['sub'])/count['reads'],
            'ins_per_base': count['ins']/(count['reads']*args.reference_length),
            'del_per_base': count['del']/(count['reads']*args.reference_length),
            'sub_per_base': count['sub']/(count['reads']*args.reference_length),
            'total_variants_per_base': (count['ins']+count['del']+count['sub'])/(count['reads']*args.reference_length)
        }

        # print results
        print(f"Repeats: {repeats}, Reads: {count['reads']}")
        print(f"    Total insertions: {calc[repeats]['ins']}")
        print(f"    Total deletions: {calc[repeats]['del']}")
        print(f"    Total substitutions: {calc[repeats]['sub']}")
        print(f"    Total variants: {calc[repeats]['total_vars']}")
        print(f"    Insertions per read: {calc[repeats]['ins_per_read']}")
        print(f"    Deletions per read: {calc[repeats]['del_per_read']}")
        print(f"    Substitutions per read: {calc[repeats]['sub_per_read']}")
        print(f"    Total variants per read: {calc[repeats]['total_variants_per_read']}")
        print(f"    Insertions per base: {calc[repeats]['ins_per_base']}")
        print(f"    Deletions per base: {calc[repeats]['del_per_base']}")
        print(f"    Substitutions per base: {calc[repeats]['sub_per_base']}")
        print(f"    Total variants per base: {calc[repeats]['total_variants_per_base']}")

    # write output
    with open(args.output, "w") as f:

        # write header
        header = ("repeats", "reads", "ins", "del", "sub", "total_vars", "ins_per_read", "del_per_read", "sub_per_read", "total_variants_per_read", "ins_per_base", "del_per_base", "sub_per_base", "total_variants_per_base")
        f.write("\t".join(header) + "\n")        

        # write data for each number of repeats
        for repeats, count in counts.items():
            towrite = [str(calc[repeats][h]) for h in header]
            f.write("\t".join(towrite) + "\n")


if __name__ == "__main__":
    main()


