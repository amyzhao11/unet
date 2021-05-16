# unet
This contains the implementation of the original U-net for image segmentation [here](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/). This was adapted to fit the current OASIS brain MRI dimensions which are (256, 256, 1). 

## Dataset
The OASIS brain dataset is a labelled collection of human brain MRI images and segmentation masks, as part of the [OASIS brain study](https://www.oasis-brains.org/). It contains approximately 9000 preprocessed images, separated into training, validation and testing sets. Each brain MRI image is 2D and has a corresponding 2D segmentation mask.

### MRI image
The brain MRI images are grayscale with a shape of (256, 256, 1)

### MRI Segmentation Masks
The MRI segmentatiion images are also grayscale png with shape (256, 256, 1) consiting of four unique classes.

    - 0 = Background
    - 1 = CSF
    - 2 = Gray Matter
    - 3 = White Matter 

These images can then be appropriately one-hot encoded to have shape (256, 256, 4) where each pixel belongs to one of these four classes

## Architecture
This unet contains four down sampling layers followed by four up sampling layers which also contain four skip connections to retain original image resolution due to the loss of information that comes with down sampling. 
![reference image from original paper](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)
