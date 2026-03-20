# BME450-fetal_ultrasound
Github: https://github.com/shreyaKrishnan1/BME450-fetal_ultrasound/invitations

Clinical Significance:
Need to do this part

Datasets:
Fetal head and lateral ventricle dataset:
https://pmc.ncbi.nlm.nih.gov/articles/PMC10630602/
https://zenodo.org/records/8265464

Fetal head dataset for predicting mode of delivery: 
https://www.nature.com/articles/s41597-024-03266-4

Labeled dataset of fetal tumors Labels: no tumor, benign tumor, cancerous tumor 
https://www.kaggle.com/datasets/orvile/ultrasound-fetus-dataset

Review of 14 fetal ultrasound datasets (can maybe use one or multiple of these): 
https://pubmed.ncbi.nlm.nih.gov/40775990/

Fetal phantom dataset:
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10146252
https://www.kaggle.com/datasets/mdmostakinali/fpus23-an-ultrasound-fetus-phantom-dataset

Dataset of fetal head abnormalities:
https://universe.roboflow.com/hritwik-trivedi-gkgrv/fetal-brain-abnormalities-ultrasound/dataset/1


Existing Models:
One ML model on fetal ultrasounds, can work some of these ideas into our own:
https://github.com/BioMedIA-MBZUAI/FetalCLIP

Medical image segmentation models:
https://pmc.ncbi.nlm.nih.gov/articles/PMC11300773/

Models for ventriculomegaly: 
https://pmc-ncbi-nlm-nih-gov.ezproxy.lib.purdue.edu/articles/PMC12734337/

Preprocessing techniques:
LCPC transform: standardizes shapes
https://www.tsg-lab.org/image-analysis-algorithms
Other things I remember form 357:
Standardization (of color and scale)


Proposal Requirements:

Title: Deep Learning for Fetal Ultrasound Analysis to improve Ventriculomegaly Diagnoses

Team Members: Shreya Krishnan, Aditi Choudhury

Project Description:

[Clinical description of ventriculomegaly and current diagnostic techniques, and challenges]

To address this challenge, we propose to create a deep learning model which provides segmentation of the lateral ventricle (LV) and subsequently detects the presence of ventriculomegaly from fetal brain ultrasound images. This can provide a rapid workflow for medical providers to better understand whether a fetus is at risk of ventriculomegaly, improving diagnostic accuracy. Our project is split into two parts: (a) segmentation of the lateral ventricle and (b) detection of ventriculomegaly.
For part (a), we will use a dataset of 3882 annotated fetal head images, titled “Large-Scale Annotation Dataset for Fetal Head Biometry in Ultrasound Images” [A]. We will fit a TransUNet deep learning algorithm to this data in order to accomplish segmentation of the LV. The segmentation model will label each pixel in the image as “LV”, “brain not LV”, or “not brain”. Once we have established model accuracy of 90%, we will then proceed to part (B) of detecting ventriculomegaly using the segmentation of the LV completed in part (a). Additionally, from segmentation alone, physicians can analyze size of the ventricle and development of the fetus.
For part (b), we will use a subset from a dataset of 1726 images of fetal head ultrasounds labeled by fetal brain abnormality [B]. We will modify the dataset labels to be either “LV” or “not LV.” Given that there are many more non-LV images than LV images in the dataset, oversampling techniques will be used to balance both the minority and majority classes during model training. For each image, we will first apply the segmentation model from part (a) to label each pixel in the image as “LV”, “brain not LV”, or “background.” We then pass the pixel labels and the image into a classifier to classify each fetal ultrasound image as “ventriculomegaly” or “non-ventriculomegaly.”
References:
[A] https://zenodo.org/records/8265464
[B] https://universe.roboflow.com/hritwik-trivedi-gkgrv/fetal-brain-abnormalities-ultrasound/dataset/1

Max ½ page. Include information on:
What dataset you plan to use
Goal of the project



Other things that might be useful to include
Existing models/research
Clinical Significance
What types of preprocessing we could use


we could use

