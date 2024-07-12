#!/usr/bin/env python

# caclulate the maximum theoretical library size given parents and grouping

import argparse
from aavolve.utils import get_reference_name, get_parents, make_var_groups, sort_var_names

def main():
    
    args = get_args()

    # Read in parents
    parents = get_parents(args.parents)
    groups = make_var_groups(sort_var_names(parents.keys()), args.group_vars, args.group_dist)
    
    # get names of parents, including reference
    wt_name = get_reference_name(args.parents)
    parent_names = set([wt_name])
    for d in parents.values():
        for k in d.keys():
            parent_names.add(k)

    # rearrange parents to get dict with variants from each parent as values
    parents_grouped = {}
    # iterate over groups
    for group in groups:
        parents_grouped[group] = {}
        # for each group, get variants
        for parent_name in parent_names:
            parents_grouped[group][parent_name] = {parents[var][parent_name].var_id():parents[var][parent_name] for var in group if parent_name in parents[var]}
    assert len(parents_grouped) == len(groups)
    
    prod = 1
    for g in groups:
        # get number of distinct parental variants at this position
        pars = parents_grouped[g]
        g_pars = []
        for p in pars:
            if pars[p] not in g_pars:
                g_pars.append(pars[p])
        prod *= len(g_pars)

    print(f"Maximum theoretical library size: {prod:.2e}")


def get_args():

    parser = argparse.ArgumentParser(description='Calculate the maximum theoretical library size given parents and grouping')
    parser.add_argument('-p', '--parents', help='Parental variants (long format)', required=True)
    parser.add_argument("--group-vars", '-g', action="store_true", help="Group adjacent variants and use the parent with the lowest hamming distance")
    parser.add_argument("--group-dist", help="Variants separated by at most this number of nucleotides will be grouped for assigning parents", default=1, type=int)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()