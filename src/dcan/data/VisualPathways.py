class VisualPathways:
  def __init__(self, optic_radiation, meyers_loop, lateral_geniculate_body, optic_tract):
    self.optic_radiation = optic_radiation
    self.meyers_loop = meyers_loop
    self.lateral_geniculate_body = lateral_geniculate_body
    self.optic_tract = optic_tract

  def get_score(self):
    return self.optic_radiation + self.meyers_loop + self.lateral_geniculate_body + self.optic_tract
