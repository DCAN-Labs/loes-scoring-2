import csv
from datetime import datetime

from dcan.data.AnteriorTemporalWhiteMatter import AnteriorTemporalWhiteMatter
from dcan.data.AuditoryPathway import AuditoryPathway
from dcan.data.CorpusCallosum import CorpusCallosum
from dcan.data.FrontalWhiteMatter import FrontalWhiteMatter
from dcan.data.Frontopontine_And_Corticopsinal_Fibers import Frontopontine_And_Corticopsinal_Fibers
from dcan.data.LoesScore import LoesScore
from dcan.data.ParietoOccipitalWhiteMatter import ParietoOccipitalWhiteMatter
from dcan.data.VisualPathways import VisualPathways


def get_partial_loes_scores(file_path):
    loes_scores = dict()
    with open(file_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        next(csv_reader)
        for row in csv_reader:
            sub_id = row[0]
            sub_session = row[1]
            date_time_str = row[2]
            date_of_mri = datetime.strptime(date_time_str, '%m/%d/%Y')
            parieto_occipital_white_matter = \
                ParietoOccipitalWhiteMatter(
                    periventricular=float(row[3]), central=int(row[4]), subcortical=float(row[5]), atrophy=int(row[6]))
            anterior_temporal_white_matter = \
                AnteriorTemporalWhiteMatter(
                    periventricular=float(row[7]), central=float(row[8]), subcortical=float(row[9]),
                    atrophy=int(row[10]))
            frontal_white_matter = \
                FrontalWhiteMatter(
                    periventricular=float(row[11]), central=int(row[12]), subcortical=float(row[13]),
                    atrophy=float(row[14]))
            corpus_callosum = \
                CorpusCallosum(splenium=int(row[15]), body=int(row[16]), genu=int(row[17]), atrophy=int(row[18]))
            visual_pathways = \
                VisualPathways(
                    optic_radiation=int(row[19]), meyers_loop=int(row[20]), lateral_geniculate_body=int(row[21]),
                    optic_tract=int(row[22]))
            auditory_pathway = \
                AuditoryPathway(
                    medial_geniculate=float(row[23]), brachium_to_inferior_colliculus=float(row[24]),
                    lateral_leminiscus=float(row[25]), trapezoid_body_pons=float(row[26]))
            frontopontine_and_corticopsinal_fibers = \
                Frontopontine_And_Corticopsinal_Fibers(internal_capsule=float(row[27]), brain_stem=float(row[28]))
            cerebellum = int(row[29])
            white_matter_cerebellum_atrophy = int(row[30])
            basal_ganglia = int(row[31])
            anterior_thalamus = int(row[32])
            global_atrophy = int(row[33])
            brainstem_atrophy = int(row[34])
            loes_score = float(row[35])
            retricted_diffusion_present_on_mri = True if row[36] == 'Yes' else False
            gad_str = row[37]
            gad = 0.0 if gad_str == '' else float(gad_str)
            loes_score_obj = \
                LoesScore(sub_id, sub_session, date_of_mri, parieto_occipital_white_matter,
                          anterior_temporal_white_matter, frontal_white_matter, corpus_callosum, visual_pathways,
                          auditory_pathway, frontopontine_and_corticopsinal_fibers, cerebellum,
                          white_matter_cerebellum_atrophy, basal_ganglia, anterior_thalamus, global_atrophy,
                          brainstem_atrophy, loes_score, retricted_diffusion_present_on_mri, gad)
            if sub_id not in loes_scores:
                loes_scores[sub_id] = dict()
            loes_scores[sub_id][sub_session] = loes_score_obj
    return loes_scores
