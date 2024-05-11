# FA-Net: A Fuzzy Attention-aided Deep Neural Network for Pneumonia Detection in Chest X-Rays
This is the official implementation  of "FA-Net: A Fuzzy Attention-aided Deep Neural Network for Pneumonia Detection in Chest X-Rays" (IEEE CBMS, 2024)

### Overall workflow:
![architecture](https://github.com/AyushRoy2001/FA-Net/assets/94052139/61446df1-e121-4d63-b082-38a3d5845cc5)

##  Channel Attention Module (CAM):
![CAM](https://github.com/AyushRoy2001/FA-Net/assets/94052139/3625925c-4bcf-47b0-ae3f-a4bc14c7bdf9)

##  Fuzzy Channel Selection (FCS):
![FCS](https://github.com/AyushRoy2001/FA-Net/assets/94052139/b9855873-e2e7-4fc9-abae-e78d17f6f01d)

## How to use
-Fork the repository.<br/>
-Download the Kermany et al. pneumonia classification dataset and store each class in a separate folder named "class_name".<br/>
-Augment the dataset (rotation, noise addition, cropping).<br/>
-Run the jupyter notebook to train the model and generate results. Make sure to change the paths according to your requirement.<br/>

## Results
### Heat maps
![heatmap](https://github.com/AyushRoy2001/FA-Net/assets/94052139/4f12bad6-6fd2-4c9d-b56f-f1286fb22ab5)

### Confusion matrix 
![cm](https://github.com/AyushRoy2001/FA-Net/assets/94052139/5fcf8eeb-a6a4-4a5a-b4c5-a99854eea05d)

### Subspace feature representation
![space](https://github.com/AyushRoy2001/FA-Net/assets/94052139/75a6ac14-29f5-4a54-a31e-c5174d296ed2)

## Authors :nerd_face:*
Ayush Roy<br/>
Anurag Bhattacharjee<br/>
Ram Sarkar<br/>
Diego Oliva<br/>
Oscar Ramos-Soto<br/> 
Francisco J. Alvarez-Padilla<br/>
