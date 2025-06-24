                                                                                                                  VISON AI SUIT CLEANED CODES

CAPTIONING MODEL
dataset.py - importing and organizing data
data_preprocessing - preprocessing data
encoder.py, decoder.py - models for image captioning
train.py - initial trainingof captioning model
check_vocab_and_weights.py - this script helps to check the if wrod2idx = idx2word.
finetune5.py - finetuning with augmented dataand also by adjusting the learning rates.
inference.py - inference script with bean search for infering the captioning model.
evaluate.py - this script helps to evaluate the captioning model performance using Bleu and Meteor matrices.
app.py - this script is used for partial visualization of the captioning model.

SEGMENTATION MODEL
s_datasets.py - this script is used for the dataset preprocessing for U-net segmentation model.
U-net.py -  this script defines the model architecture.
s_train.py - this script trains the U-net model with developed architecture and pretrained weights.
s_finetune.py - this script is used for finetuning the U-net model.
utils - this script refers to the utilities of the segmentation model.
s_app.py - this scipt helps in visalization of the semantic segmentation model.

PIPELINING
pipeine.py - this script connects both the captioning model and the segmentation model together and also implements two more new models :
              a. Mask R-CNN - for instance segmentation .
              b. faster R-CNN - for object detection.




