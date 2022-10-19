import os.path
from datetime import datetime
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np

from dcan.data.partial_loes_scores import get_partial_loes_scores

fig, axs = plt.subplots(4, 6)
fig.suptitle('Loes score progression')
scores_list = []
loes_scoring_folder = '/home/miran045/reine097/projects/loes-scoring-2/data'
partial_scores_file_path = os.path.join(loes_scoring_folder, '9_7 MRI sessions Igor Loes score updated.csv')
partial_loes_scores = get_partial_loes_scores(partial_scores_file_path)
doc_folder = '/home/miran045/reine097/projects/loes-scoring-2/doc'
n = 11
colors = plt.cm.rainbow(np.linspace(0, 1, n))
with open(os.path.join(doc_folder, 'LoesScoreProgression.md'), 'w') as the_file:
    for subject in partial_loes_scores:
        subject_scores = []
        print(subject)
        subject_data = partial_loes_scores[subject]
        t = []
        scores = []
        anterior_temporal_white_matter_scores = []
        anterior_thalamus_scores = []
        auditory_pathway_scores = []
        basal_ganglia_scores = []
        cerebellum_scores = []
        corpus_callosum_scores = []
        frontal_white_matter_scores = []
        frontopontine_corticalspinal_fibers_scores = []
        parieto_occipital_white_matter_scores = []
        visual_pathways_scores = []
        for session in subject_data:
            parts = session.split('_')
            session_date_str = parts[1]
            datetime_object = datetime.strptime(session_date_str, '%Y%m%d')
            t.append(datetime_object)
            session_data = subject_data[session]
            loes_score = session_data.loes_score
            anterior_temporal_white_matter_score = session_data.anterior_temporal_white_matter.get_score()
            anterior_temporal_white_matter_scores.append(anterior_temporal_white_matter_score)
            anterior_thalamus_score = session_data.anterior_thalamus
            anterior_thalamus_scores.append(anterior_thalamus_score)
            auditory_pathway_score = session_data.auditory_pathway.get_score()
            auditory_pathway_scores.append(auditory_pathway_score)
            basal_ganglia_score = session_data.basal_ganglia
            basal_ganglia_scores.append(basal_ganglia_score)
            cerebellum_score = session_data.cerebellum
            cerebellum_scores.append(cerebellum_score)
            corpus_callosum_score = session_data.corpus_callosum.get_score()
            corpus_callosum_scores.append(corpus_callosum_score)
            frontal_white_matter_score = session_data.frontal_white_matter.get_score()
            frontal_white_matter_scores.append(frontal_white_matter_score)
            frontopontine_corticalspinal_fibers_score = session_data.frontopontine_and_corticopsinal_fibers.get_score()
            frontopontine_corticalspinal_fibers_scores.append(frontopontine_corticalspinal_fibers_score)
            parieto_occipital_white_matter_score = session_data.parieto_occipital_white_matter.get_score()
            parieto_occipital_white_matter_scores.append(parieto_occipital_white_matter_score)
            visual_pathways_score = session_data.visual_pathways.get_score()
            visual_pathways_scores.append(visual_pathways_score)
            sum_sub_scores = anterior_temporal_white_matter_score + anterior_thalamus_score + auditory_pathway_score + basal_ganglia_score + cerebellum_score + corpus_callosum_score + frontal_white_matter_score + frontopontine_corticalspinal_fibers_score + parieto_occipital_white_matter_score + visual_pathways_score
            # assert abs(sum_sub_scores - loes_score) < 0.1
            scores.append(loes_score)
            print(f'\t{datetime_object}: {loes_score}')
            subject_scores.append((datetime_object.timestamp(), loes_score))
        scores_list.append(subject_scores)
        fig, ax = plt.subplots()
        ax.plot(t, scores, color=colors[0], label='total')
        ax.plot(t, anterior_temporal_white_matter_scores, color=colors[1], label='anterior_temporal_white_matter')
        ax.plot(t, anterior_thalamus_scores, color=colors[2], label='anterior_thalamus')
        ax.plot(t, anterior_thalamus_scores, color=colors[3], label='auditory_pathway')
        ax.plot(t, basal_ganglia_scores, color=colors[4], label='basal_ganglia')
        ax.plot(t, basal_ganglia_scores, color=colors[5], label='cerebellum')
        ax.plot(t, basal_ganglia_scores, color=colors[6], label='corpus_callosum')
        ax.plot(t, frontal_white_matter_scores, color=colors[7], label='frontal_white_matter')
        ax.plot(t, frontopontine_corticalspinal_fibers_scores, color=colors[8], label='frontopontine_corticalspinal_fibers')
        ax.plot(t, parieto_occipital_white_matter_scores, color=colors[9], label='parieto_occipital_white_matter')
        ax.plot(t, visual_pathways_scores, color=colors[10], label='visual_pathways')
        ax.set_title(subject)
        plt.xlabel('Session date')
        plt.xticks(rotation=30)
        plt.ylabel('Loes score')
        plt.legend()
        image_name = f'{subject}.png'
        the_file.write(f'![](./img/{image_name})\n\n')
        plt.savefig(os.path.join(doc_folder, 'img', image_name))
print(scores_list)
