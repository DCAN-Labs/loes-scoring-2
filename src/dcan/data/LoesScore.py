class LoesScore:
    def __init__(self, sub_id, sub_session, date_of_mri, parieto_occipital_white_matter, anterior_temporal_white_matter,
                 frontal_white_matter, corpus_callosum, visual_pathways, auditory_pathway,
                 frontopontine_and_corticopsinal_fibers, cerebellum, white_matter_cerebellum_atrophy, basal_ganglia,
                 anterior_thalamus, global_atrophy, brainstem_atrophy, loes_score, retricted_diffusion_present_on_mri,
                 gad):
        self.sub_id = sub_id
        self.sub_session = sub_session
        self.date_of_mri = date_of_mri
        self.parieto_occipital_white_matter = parieto_occipital_white_matter
        self.anterior_temporal_white_matter = anterior_temporal_white_matter
        self.frontal_white_matter = frontal_white_matter
        self.corpus_callosum = corpus_callosum
        self.visual_pathways = visual_pathways
        self.auditory_pathway = auditory_pathway
        self.frontopontine_and_corticopsinal_fibers = frontopontine_and_corticopsinal_fibers
        self.cerebellum = cerebellum
        self.white_matter_cerebellum_atrophy = white_matter_cerebellum_atrophy
        self.basal_ganglia = basal_ganglia
        self.anterior_thalamus = anterior_thalamus
        self.global_atrophy = global_atrophy
        self.brainstem_atrophy = brainstem_atrophy
        self.loes_score = loes_score
        self.retricted_diffusion_present_on_mri = retricted_diffusion_present_on_mri
        self.gad = gad
