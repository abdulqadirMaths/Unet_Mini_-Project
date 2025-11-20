Simple U-Net Segmentation Project

This is my small UNet project that I made for learning purpose. I used Google Colab to write all the code and then uploaded it here.
The project is about image segmentation where the model takes an image and predicts a mask.

What this project includes

Data loading code

Building a small UNet model

Training the model

Plotting predictions

I mostly followed tutorials and tried to understand the architecture.

How to run

You can open the notebook from notebooks/ folder.
Just run all the cells one by one (I did it in Google Colab).
Make sure the dataset is inside the data/ folder.

Files

notebooks/unet_training.ipynb → whole code

src/model.py → UNet model code

src/train.py → training steps

src/plot_results.py → plotting function

data/ → put your images and masks here

What I learned

How UNet works

How to train simple segmentation model

How to upload project on GitHub