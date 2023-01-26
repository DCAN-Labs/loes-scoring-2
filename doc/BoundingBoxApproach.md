Computing Loes Scores with Bounding Boxes Determined by Deep Learning
=====================================================================

This is a proposal for an alternative deep learning strategy for computing
Loes scores using bounding 'boxes' (cuboids, actually).

Phase 1
-------

Manually draw ground-truth bounding boxes around the areas that have a 1 score.  
Limit the bounding box to the area affected by adrenal leukodystrophy (ALD), not the whole anatomical region.  We 
won't actually draw the boxes but add the vertices of the bounding boxes to a 
spreadsheet.  (There 
might be a way to semi-automate this in Python, so the user can 'draw' on an MRI 
image.)
This is time-consuming, but as a side-effect might give us other insights 
helpful for computing Loes scores.  This could be done in one or more hackathons.

Phase 2
-------

Create a deep learning model that finds bounding boxes of regions affected by adrenal
leukodystrophy.  Count the number of such boxes and use this as a surrogate
for the Loes score or as an input to another deep learning model for computing Loes scores.
The general approach for this is defined in 
["TensorFlow 2 meets the Object Detection API"](https://blog.tensorflow.org/2020/07/tensorflow-2-meets-object-detection-api.html).
A specific example was an exercise I did, 
["Zombie Detection"](https://colab.research.google.com/drive/1c6ROJZ4aG9mtcXqHpk3COhrzkHtVL1qa?usp=sharing).
(Seriously.)