# Neural Style Transfer

This is a PyTorch implementation of neural style transfer that I made for practice and for fun. It aims to reproduce the results from the papers: 
* [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (Leon A. Gatys, Alexander S. Ecker, Matthias Bethge)
* [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) (Leon A. Gatys, Alexander S. Ecker, Matthias Bethge)

Neural style transfer is an optimization technique used to take three images, a content image, a style reference image (such as an artwork by a famous painter), and the input image you want to style — and blend them together such that the input image is transformed to look like the content image, but “painted” in the style of the style image. 

## Results

**Style**: Femme nue assise by Pablo Picasso, 1910
**Content**: Photo of the Neckarfront, Tuebingen (Germany)
![picasso result](results/picasso.gif)

**Style**: Composition VII by Wassily Kandinsky, 1913
**Content**: Photo of the Neckarfront, Tuebingen (Germany)
![picasso result](results/comp.gif)

**Style**: [Logo](https://1000logos.net/wp-content/uploads/2016/10/Barcelona-symbol.jpg) of FC Barcelona
**Content**: Photo I took at Camp Nou, Barcelona (Spain)
![picasso result](results/campnou.gif)

## Usage

```
usage: Neural Style Transfer [-h] [--output_folder OUTPUT_FOLDER] [--gif]
                             [--alpha ALPHA] [--epochs EPOCHS]
                             [--img_size IMG_SIZE] [--device DEVICE]     
                             [--log_interval LOG_INTERVAL]
                             content style

positional arguments:
  content               Path to the content reference image
  style                 Path to the style reference image

optional arguments:
  --output_folder       Folder to save generated images. Default: Folder named
                        'generated' in cwd.
  --gif                 If argument specified, additionally create gif from
                        generated images
  --alpha ALPHA         Ratio of style to content when computing loss. Use
                        larger value to capture more style. Default: 10000.
                        Typical value: between 100 and 100000
  --epochs EPOCHS       Number of iterations. Default: 300
  --img_size IMG_SIZE   Width of output image. Aspect ratio is set to 4:3.
                        Default: 640
  --device DEVICE       'cuda' for GPU, 'cpu' for CPU. By default, selects GPU
                        if available
  --log_interval LOG_INTERVAL
                        Number of iterations after which to save image.
                        Default: 50
```

For example:
```
python style.py references/tubingen.jpg references/picasso.jpg --alpha 100000 --epochs 500 --gif --log_interval 10
```