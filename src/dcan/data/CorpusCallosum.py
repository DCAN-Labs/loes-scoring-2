class CorpusCallosum:
  def __init__(self, splenium, body, genu, atrophy):
    self.splenium = splenium
    self.body = body
    self.genu = genu
    self.atrophy = atrophy

  def get_score(self):
    return self.splenium + self.body + self.genu
