# U-Net Based Brain Segmentation

Report link [U-Net Based Brain Segmentation](https://github.com/yanlong-sun/brain_segmentation/blob/master/Report.pdf)

## Usage 
### Predicting  
-  Run `Input.m`    
     `../test_data/`  -  folder of original 'nii.gz' files    
     `./data/test/`    -  folder of preprocessed slices   
  
-  Open `brain_segmentation.ipynb` In Google Colaboratory (python == 2.x)    
	```
  	! python pred.py 
	```
-  Run `save_result.m`  
	`./predictions` -  folder of prediction  
  
### Training model
-  Run `Input.m`  
     `../training_data/`  -  folder of original 'nii.gz' files   
     `../valid_data/`  -  folder of original 'nii.gz' files   
     `./data/train/`    -  folder of preprocessed slices  
     `./data/valid/`    -  folder of preprocessed slices  
  
-  Open `brain_segmentation.ipynb` In Google Colaboratory (python == 2.x)   
	```
  	! python train.py   
	```
*  We got `weights_128.h5` from training on  "CC-359"      CC0001 - CC0050   
* valid  CC0050 - CC0060
	  	
		  

## Results
Here we show some outputs and qualitative results of our test subjects.
![image](https://github.com/yanlong-sun/brain_segmentation/blob/master/report/1.png)
![image](https://github.com/yanlong-sun/brain_segmentation/blob/master/report/2.png)
![image](https://github.com/yanlong-sun/brain_segmentation/blob/master/report/3.png)
![image](https://github.com/yanlong-sun/brain_segmentation/blob/master/report/DSC.png)

## Image data
[Calgary-Campinas-359 (CC-359) dataset](https://sites.google.com/view/calgary-campinas-dataset/download?authuser=0)