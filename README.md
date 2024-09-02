# PainterMcgee
PainterMcgee is a repository developed for our AI final project, where we explore various AI-driven approaches to approximate target images using rectangles. 
This project employs three distinct algorithms to tackle the problem, each with its unique methodology and strengths.

# Overview of Approaches
We implemented the following three models to approximate images:

**Genetic Algorithm:** <br />
-Mimics the process of natural selection to iteratively improve the image approximation. <br />
-To run this model, execute the script: genetic_boi.py.

**Local Beam Search:** <br />
-Explores multiple paths in parallel, focusing on the most promising candidates. <br />
-To run this model, execute the script: beam_search.py. 

**MCTS:** <br />
-Uses simulations to explore possible moves and select the best strategy based on accumulated results. <br />
-To run this model, execute the script: monte_carlo.py.

# How to Use
Each script allows customization through various hyperparameters defined at the top of the file. <br />
You can adjust these parameters to experiment with different configurations and observe how they affect the image approximation process.

# Setting the Target Image
The target image that each model will attempt to approximate is specified by the following parameter at the top of each file: <br />
IMAGE_NAME = 'image_name' <br />
The image file should be placed in the layouts directory, with the path formatted as layouts/{IMAGE_NAME}.jpg.

# Using Your Own Images
If you'd like to use a different target image: <br />
Upload your image to the layouts directory as a .jpg file. <br />
Update the IMAGE_NAME parameter in the script to match your file name (without the .jpg extension). <br />

We have included some pre-loaded images in the layouts directory that you can use to test the models. <br />
Feel free to experiment with these or upload your own images following the instructions above.



