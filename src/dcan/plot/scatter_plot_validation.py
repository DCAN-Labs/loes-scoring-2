import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

loes_scores_file = '/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/loes_scores_all_predictions.csv'
loes_scores_df = pd.read_csv(loes_scores_file)
loes_scores_df = loes_scores_df.reset_index()
ratings_dict = dict()
xs = []
ys = []
for index, row in loes_scores_df.iterrows():
    xs.append(row['loes_score'])
    ys.append(row['prediction'])
loes_scores_df.drop(['level_0', 'subject', 'First Name', 'Last Name', 'Date of Birth', 'Date of MRI', 'session'], inplace=True, axis=1)
loes_scores_df.to_csv('/home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/anonymized_loes_scores_all_predictions_out.csv', index=False)

fig, ax = plt.subplots()
plt.scatter(xs, ys)
plt.title('Actual Loes score vs. predicted Loes score')
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

# now plot both limits against each other
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.xlabel("Actual Loes score")
plt.ylabel("Predicted Loes score")

output_file = '/home/miran045/reine097/projects/loes-scoring-2/doc/img/results.png'
plt.savefig(output_file)
plt.show()
