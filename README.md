# Skullbinarymask
uNET model for creating binary masks of skulls for 3D photogrammetry. The model you want to use is unet_pytorch_split_model.pth
Train from scratch using unettorchnosplit.py
Save masks created to a folder using saveallmasks.py
Use modeltesttorch.py to test the model's prediction on a single image and see if that's what you need. Change the directories in lines 11 and 12 appropriately.

Edit lines 13 and 14 in saveallmasks.py to change the input and output directories respectively. The program is set up to use CUDA if set up but will default to using CPU if not. If you're using an AMD (or other) GPU, feel free to change line 15 to the correct device. Once the virtual environment is set up with the dependencies listed in requirements.txt, you should be good to go! Feel free to add issues in if you face problems with the install and usage of this program.

This Program was developed at the Research lab of Dr. Duncan Irschick at the University of Massachusetts - Amherst. The mammal skulls used to train this model were sourced from the Mammal collection at the University of Massachusetts - Amherst (Big Thank You to Kate Doyle). It was used as a part of a research project on the analysis of Mammalian Skull shape led by myself and Owen Atkins.

This should just work on any lighter colored object on a solid dark background. 
The primary use for this was to create masks for Agisoft Metashape in the production of 3D models of mammalian skulls, so it definitely works on bones :) The training data set was comprised of like 2000 images of bones and their masks what were edited in a photo editing software. It was trained for 5 epochs and reached a 98.8% acccuracy. If you manage to add to the dataset and train a better model, please reach out because I wanna see!!!

I'm sure there are ways to do this that are either bulkier or cleaner but this is my solution. LLMs are too wasteful and this could probably run on a chromebook if you tried hard enough (has not been verified, do not blame me if your chromebook catches fire). My benchmark is a 5 year old AMD ryzen 7 5800H and 16 GB RAM, to generate 20 masks in a minute using the model listed in the repo. The model was trained on a desktop 4070Ti and 32GB RAM and took like 4 hours. 16GB should be enough to train through unettorchnosplit.py but it is dicey and takes forever. 

p.s. I tried using tensorflow and........ no. 
p.p.s Please reach out if you need some help with this, I'd be happy to assist :).
p.p.p.s Not that I believe anybody is going to bother to sell this, but it was created to be open source as part of a research project so everybody can use it for free!


