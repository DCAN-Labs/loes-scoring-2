class Frontopontine_And_Corticopsinal_Fibers:
  def __init__(self, internal_capsule, brain_stem):
    self.internal_capsule = internal_capsule
    self.brain_stem = brain_stem

  def get_score(self):
    return self.internal_capsule + self.brain_stem
