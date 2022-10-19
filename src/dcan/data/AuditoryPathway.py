class AuditoryPathway:
  def __init__(self, medial_geniculate, brachium_to_inferior_colliculus, lateral_leminiscus, trapezoid_body_pons):
    self.medial_geniculate = medial_geniculate
    self.brachium_to_inferior_colliculus = brachium_to_inferior_colliculus
    self.lateral_leminiscus = lateral_leminiscus
    self.trapezoid_body_pons = trapezoid_body_pons

  def get_score(self):
    return self.medial_geniculate + self.brachium_to_inferior_colliculus + self.lateral_leminiscus + self.trapezoid_body_pons