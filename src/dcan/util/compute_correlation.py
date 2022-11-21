import ast

import scipy
import scipy.stats


def compute_pearson_correlation_coefficient():
    input_file = '/home/miran045/reine097/projects/loes-scoring-2/doc/distributions/model01.txt'
    with open(input_file) as f:
        data = f.read()

        d = ast.literal_eval(data)

        xs = []
        ys = []
        for x in d:
            vals = d[x]
            for y in vals:
                xs.append(x)
                ys.append(y)
        result = scipy.stats.linregress(xs, ys)

        return result

result = compute_pearson_correlation_coefficient()

print(f"correlation:    {result.rvalue}")
print(f"p-value:        {result.pvalue}")
print(f"standard error: {result.stderr}")
