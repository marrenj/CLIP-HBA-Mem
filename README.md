<h1 align="center"> 
    <img src="./CLIP-HBA/figures/hba_logo.png" width="400">
</h1>



<h1 align="center">
    <p> Shifting Attention to You: Personalized Brain-Inspired AI Models <br></p>
</h1>

<h3 align="center">
    <p> Tovar Lab @ Vanderbilt University <br></p>
</h3>


# Overview

This repository contains all the key training, inference, and visualization script for behavioral/meg fine-tuning with the CLIP architecture. 

# Abstract

The integration of human and artificial intelligence represents a scientific opportunity to advance our understanding of information processing, as each system offers unique computational insights that can enhance and inform the other. The synthesis of human cognitive principles with artificial intelligence has the potential to produce more interpretable and functionally aligned computational models, while simultaneously providing a formal framework for investigating the neural mechanisms underlying perception, learning, and decision-making through systematic model comparisons and representational analyses. In this study, we introduce personalized brain-inspired modeling that integrates human behavioral embeddings and neural data to align with cognitive processes. We took a step-wise approach where we fine-tuned the Contrastive Language–Image Pre-training (CLIP) model with a large scale behavioral decisions, group-level neural data, and finally participant-level neural data in a broader framework we have named CLIP-Human Based Analysis (CLIP-HBA). We found that the model fine tuned on a static behavioral embedding (CLIP-HBA-Behavior) significantly enhances its ability to predict human similarity judgments while indirectly aligning it with dynamic representations captured via magnetoencephalography (MEG). To further gain mechanistic insights into the evolution of cognitive processes, we introduced a model specifically fine-tuned on millisecond-level MEG neural dynamics (CLIP-HBA-MEG). This model resulted in enhanced temporal alignment with human neural processing while still showing improvement on behavioral alignment. Finally, we trained individualized models on participant-specific neural data, effectively capturing unique neural dynamics and highlighting the potential for personalized AI systems. Our findings integrate human-based representations into AI development, advancing personalized AI and precise quantification of individual perceptual differences. 


# Code Structure: 

```
───CLIP-HBA # Main Folder for training, inference, and visualization
│   ├───Brain_Alignments # Visualizations for model-neural alignment with various dataset
│   │   └───Hebart1854_model_rdms
│   ├───Data # required images, arrays, and annotations for training and inference
│   │   ├───Cichy # Individual Training Data Source
│   │   │   └───stimuli
│   │   ├───Encoder_Correspondence # Encoder Correspondence for the model (used for matrix pre-optimization)
│   │   │   └───weighting_matrix
│   │   ├───figures
│   │   ├───misc # most miscellaneous data EDA and visualization scripts
│   │   │   └───temp_rdms
│   │   ├───models
│   │   │   ├───partial
│   │   ├───test_images
│   │   ├───Things1854
│   │   ├───ThingsMEG_RDMs
│   │   └───TovarDataset
│   │       ├───...
│   ├───figures # figures used for this README and demonstrations
│   └───sizesearch # search the minimum size of training set for effective behavioral training
├───mmedit # model backbone scripts
│   ├─── ...
│   ├───models
│   │   ├───backbones
│   │   │   ├───encoder_decoders
│   │   │   │   ├───decoders
│   │   │   │   │   
│   │   │   │   ├───encoders
│   │   │   │   │   
│   │   │   │   ├───necks
│   │   │   │   │   
│   │   │   ├───generation_backbones
│   │   │   │   
│   │   │   ├───sr_backbones
│   │   │   │   
│   │   │   ├───vfi_backbones
│   │   │   │   
│   │   ├───...
│   │   ├───components
│   │   │   ├───clip
│   │   │   │   
│   │   │   ├───...
│   │   │   │   
│   │   │   ├───clip_hba_dynamic_3d # Dynamic CLIP-HBA-MEG (Group & Individual)
│   │   │   │   
│   │   │   ├───clip_hba_no_softmax # Static CLIP-HBA-Behavior
│   │   │   │   
│   │   │   ...

│   ├───utils
│   │   
│   
└───output # output of the model inference scripts
```

# Environment Setup:

```
conda create -n cliphba python=3.8.18
conda activate cliphba

conda install pytorch=1.10.2 torchvision cudatoolkit=11.3 -c pytorch

pip install openmim
mim install mmcv-full==1.5.0
pip install -r requirements.txt
```

download the pretrained CLIP-HBA model weights from [here](https://drive.google.com/file/d/1_X9w3ttJt419gb8hosBbHSbG3WwRQ2GW/view?usp=share_link)


# Behavioral Training - Things Dataset
```
cd ./CLIP-HBA
python train_behavior_things.py #Change the dataset path and model configurations in the train.py file
```


# MEG Group Level Training - Things MEG Data with 3 Participants
```
cd ./CLIP-HBA
python train_meg_things.py #Change the dataset path and model configurations in the train_dynamic.py file
```

# MEG Group Level Training - 118 Images with 15 Participants
```
cd ./CLIP-HBA
python train_individual_cichy.py #Change the dataset path and model configurations in the train_dynamic.py file
```

# Inference - Behavior/Static Embeddings and RDMs
```
cd ./CLIP-HBA
python inference_behavior.py # Change the image path and model configurations in the inference.py file
```

# Inference - MEG/Dynamic Embeddings and RDMs 
```
cd ./CLIP-HBA
python inference_meg.py # Change the image path and model configurations in the inference.py file
```



# Opensourced Training Data

## THINGS
Download Things-MEG Data from [here](https://plus.figshare.com/articles/dataset/THINGS-data_MEG_preprocessed_dataset/21215246?backTo=/collections/THINGS-data_A_multimodal_collection_of_large-scale_datasets_for_investigating_object_representations_in_brain_and_behavior/6161151)
s

## Things Image Set
Things Things Image Dataset can be downloaded from [here](https://osf.io/jum2f/)
