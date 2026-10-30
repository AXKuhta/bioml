from bisect import bisect_left

spans_dna = []
spans_rna = []

def read_ints(itr):
	return [int(x) for x in next(itr).split(" ")]

with open(0) as f:
	n_dna, n_rna = read_ints(f)

	for i in range(n_dna):
		spans_dna.append(read_ints(f))

	for i in range(n_rna):
		spans_rna.append(read_ints(f))

def count_reads(spans_dna, spans_rna):
    n_genes = len(spans_dna)
    n_reads = len(spans_rna)

    # Each row contains endpoint pairs, just as in the original code.
    genes = [
        (row[j], row[j + 1], gene_id)
        for gene_id, row in enumerate(spans_dna)
        for j in range(0, len(row) - 1, 2)
    ]

    # Sort RNA segments by their right endpoint.
    reads = [
        (row[j + 1], row[j], read_id)
        for read_id, row in enumerate(spans_rna)
        for j in range(0, len(row) - 1, 2)
    ]

    ans = [0] * n_genes
    if not genes or not reads:
        return ans

    genes.sort()
    reads.sort()

    ends = sorted({end for _, end, _ in genes})
    size = len(ends)

    # Reverse endpoint ranks so "end >= start" becomes a prefix query.
    end_rank = {
        end: size - i
        for i, end in enumerate(ends)
    }

    tree_min = [n_genes] * (size + 1)
    tree_max = [-1] * (size + 1)

    read_min = [n_genes] * n_reads
    read_max = [-1] * n_reads

    p = 0

    for read_end, read_start, read_id in reads:
        # Activate every gene segment whose start <= this read's end.
        while p < len(genes) and genes[p][0] <= read_end:
            _, gene_end, gene_id = genes[p]
            k = end_rank[gene_end]

            while k <= size:
                tree_min[k] = min(tree_min[k], gene_id)
                tree_max[k] = max(tree_max[k], gene_id)
                k += k & -k

            p += 1

        # Among activated segments, query those whose end >= read_start.
        k = size - bisect_left(ends, read_start)
        lowest = n_genes
        highest = -1

        while k > 0:
            lowest = min(lowest, tree_min[k])
            highest = max(highest, tree_max[k])
            k -= k & -k

        # Combine hits across all segments belonging to this RNA read.
        read_min[read_id] = min(read_min[read_id], lowest)
        read_max[read_id] = max(read_max[read_id], highest)

    for lowest, highest in zip(read_min, read_max):
        if lowest == highest:
            ans[lowest] += 1

    return ans

for count in count_reads(spans_dna, spans_rna):
    print(count)
