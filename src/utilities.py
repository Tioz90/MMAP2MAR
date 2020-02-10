import math


def entropy(pmf, base = 2):
	e = 0.0
	if max(pmf) < 1.0:
		for p in pmf:
			if p > 0:
				e -= p*math.log(p, base)
	return e