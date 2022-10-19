class FrontalWhiteMatter:
  def __init__(self, periventricular, central, subcortical, atrophy):
    self.periventricular = periventricular
    self.central = central
    self.subcortical = subcortical
    self.atrophy = atrophy

  def get_score(self):
    return self.periventricular + self.central + self.subcortical
