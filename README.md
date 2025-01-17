setup environment:

```
conda create -n cliphba python=3.8.18
conda activate cliphba

conda install pytorch=1.10.2 torchvision cudatoolkit=11.3 -c pytorch

pip install openmim
mim install mmcv-full==1.5.0
pip install -r requirements.txt
```

download the pretrained CLIP-HBA model weights from [here](https://drive.google.com/file/d/1_X9w3ttJt419gb8hosBbHSbG3WwRQ2GW/view?usp=share_link)


# Behavioral Embedding Training
```
cd ./CLIP-HBA
python train.py #Change the dataset path and model configurations in the train.py file
```

# MEG Dynamic Training
```
cd ./CLIP-HBA
python train_dynamic.py #Change the dataset path and model configurations in the train_dynamic.py file
```

# Inference
```
cd ./CLIP-HBA
python inference_vit.py # Change the image path and model configurations in the inference.py file
```

# RDM Generation after inference embeddings are produced
This code takes the inference model output embeddings of any images and generate rdms in h5 format. 
```
cd ./CLIP-HBA
python rdm_generation_clip.py # Change the embedding path and saving path. 
```

# Hebart Brain Alignment
```
./CLIP-HBA/Brain_Alignments/hebart_brain_alignments.ipynb # for Hebart Things 1854 Objets MEG Alignment
```
Download Things-MEG Data from [here](https://plus.figshare.com/articles/dataset/THINGS-data_MEG_preprocessed_dataset/21215246?backTo=/collections/THINGS-data_A_multimodal_collection_of_large-scale_datasets_for_investigating_object_representations_in_brain_and_behavior/6161151)

# External Brain Alignment
See the `Brain_Alignments` folder for the code to align the embeddings with the brain data.

# Output folder
All inference model embedding outputs are stored in the `output` folder.
`hba_dynamic` and `vit_dynamic` folders contain inference embeddings of models that were trained with Things 1854 dataset with two training configurations: `dynamic48` means the model was trained with 48 images and `dynamic1806` means the model was trained with 1806 images. Same naming convention applies throughout this repo. 

# Things Image Set
Things Things Image Dataset can be downloaded from [here](https://osf.io/jum2f/)