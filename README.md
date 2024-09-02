# PainterMcgee
This repository contains the codebase for our AI final project, in which we use models to approximate images using rectangles.

We approached the problem using three different models:

Genetic algorithm - which can be run by running genetic_boi 

Local beam search - which can be run by running beam_search

MCTS - which can be run by running monte_carlo

At the top of each file one can change the different hyper-parameters that specific model uses

The target image the model will try to approximate is determined by this parameter at the top of each file:
IMAGE_NAME = 'image_name' 

The image is then taken as 'layouts/{IMAGE_NAME}.jpg' 

Notice we already uploaded some images we experimented with to layouts, if one wants to use different images,
upload to layouts as a jpg and change IMAGE_NAME. 


