import bisect

spans_dna = []
spans_rna = []

def read_ints(itr):
	return [int(x) for x in next(itr).split(" ")]

with open("tests/4") as f:
	n_dna, n_rna = read_ints(f)

	for i in range(n_dna):
		spans_dna.append(read_ints(f))

	for i in range(n_rna):
		spans_rna.append(read_ints(f))

def endstops(x):
	sentinel = object()
	itr = iter(x)

	while True:
		u = next(itr, sentinel)
		v = next(itr, sentinel)

		if u == sentinel or v == sentinel:
			return
		else:
			yield u, v

#print(len(spans_dna), "genes")
#print(len(spans_rna), "reads")

ans = [0]*len(spans_dna)

for expr in spans_rna:
	hits = dict()
	for i, gene in enumerate(spans_dna):
		for enter, leave in endstops(gene):
			for _enter, _leave in endstops(expr):
				if leave >= _enter >= enter:
					hits[i] = 1
				elif leave >= _leave >= enter:
					hits[i] = 1
				elif leave >= _leave and enter <= _enter:
					hits[i] = 1
				elif _leave >= leave and _enter <= enter:
					hits[i] = 1

	# Expressed in one gene?
	if len(set(hits)) == 1:
		k, v = next(iter(hits.items()))
		ans[k] += v

for x in ans:
	print(x)
