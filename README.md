## Title 
Deep Learning for Fetal Ultrasound Analysis to improve Diagnosis of Fetal Lateral Ventriculomegaly
## Team Members
Shreya Krishnan (shreyakrishnan1), Aditi Choudhury (adi-c23)
## Project Description
Fetal lateral ventriculomegaly (FLV) is a condition where the lateral ventricles are enlarged in utero, diagnosed primarily by ultrasound of the fetal head during pregnancy [1]. FLV results in a wide range of neurodevelopmental outcomes, including motor and cognitive impairment [2]. Typically, FLV is diagnosed when the lateral ventricle size is greater than 10 mm [1]. Diagnosis requires identification of the lateral ventricle on ultrasound, measurement, and classification. Manual segmentation of the medical image can be time-consuming, especially since it requires clinical expertise [3]. 

We propose a deep learning model workflow to address challenges in FLV diagnosis. This workflow is split into two parts: (A) automated segmentation of the lateral ventricle and (B) detection of FLV. This workflow can provide rapid insights for medical providers, improving diagnostic accuracy for lateral ventriculomegaly.

For part (A), a dataset of 3882 annotated fetal head ultrasound images will be used [4]. A TransUNet algorithm will be fit to this dataset to segment the lateral ventricle. The model will assign a label to each pixel in the image to indicate whether it is: (i) part of lateral ventricles, (ii) part of the brain but not lateral ventricles, or (iii) not part of the brain. Once model accuracy of 80% is established, the segmentation will be used in part (B) to detect FLV.

For part (B), a dataset of 1726 images of fetal head ultrasounds labeled by fetal brain abnormality will be used [5]. The abnormality labels will be modified into a binary categorical variable with classes of (i) FLV and (ii) not FLV. Given that there are many more non-FLV images than FLV images in the dataset, minority oversampling will be used to balance minority and majority classes for model training. A multimodal classifier will be trained using the raw ultrasound images and segmentation results from part (A) classify if a fetal ultrasound image is indicative of lateral ventriculomegaly.


### References:
[1] A. A. Alluhaybi, K. Altuhaini, and M. Ahmad, “Fetal Ventriculomegaly: A Review of Literature,” Cureus, vol. 14, no. 2, Feb. 2022, doi: https://doi.org/10.7759/cureus.22352.

[2] V. Giorgione, K. K. Haratz, S. Constantini, R. Birnbaum, and G. Malinger, “Fetal cerebral ventriculomegaly: What do we tell the prospective parents?,” Prenatal Diagnosis, vol. 42, no. 13, pp. 1674–1681, Nov. 2022, doi: https://doi.org/10.1002/pd.6266.

[3] Farzan Vahedifard et al., “Automatic Ventriculomegaly Detection in Fetal Brain MRI: A Step-by-Step Deep Learning Model for Novel 2D-3D Linear Measurements,” Diagnostics, vol. 13, no. 14, pp. 2355–2355, Jul. 2023, doi: https://doi.org/10.3390/diagnostics13142355.

[4] M. Alzubaidi, M. Agus, M. Makhlouf, F. Anver, K. Alyafei, and M. Househ, “Large-scale annotation dataset for fetal head biometry in ultrasound images,” Data in Brief, vol. 51, p. 109708, Dec. 2023, doi: https://doi.org/10.1016/j.dib.2023.109708.

[5] H. Trivedi, “Fetal Brain Abnormalities Ultrasound Dataset,” Roboflow, 2023. https://universe.roboflow.com/hritwik-trivedi-gkgrv/fetal-brain-abnormalities-ultrasound/dataset/1 (accessed Mar. 22, 2026).
