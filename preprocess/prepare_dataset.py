import os
from pathlib import Path
from configparser import ConfigParser
import pandas as pd
import numpy as np
import warnings
import pylidc as pl
from tqdm import tqdm
from statistics import median_high
from pylidc.utils import consensus

warnings.filterwarnings(action='ignore')

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('lung.conf')

#Get Directory setting
DICOM_DIR = parser.get('prepare_dataset','LIDC_DICOM_PATH')
MASK_DIR = parser.get('prepare_dataset','MASK_PATH')
IMAGE_DIR = parser.get('prepare_dataset','IMAGE_PATH')
CLEAN_DIR_IMAGE = parser.get('prepare_dataset','CLEAN_PATH_IMAGE')
CLEAN_DIR_MASK = parser.get('prepare_dataset','CLEAN_PATH_MASK')
META_DIR = parser.get('prepare_dataset','META_PATH')
POS_NEG_RATIO = 1

#Hyper Parameter setting for prepare dataset function
mask_threshold = parser.getint('prepare_dataset','Mask_Threshold')

#Hyper Parameter setting for pylidc
confidence_level = parser.getfloat('pylidc','confidence_level')
padding = parser.getint('pylidc','padding_size')
class MakeDataSet:
    def __init__(self, LIDC_Patients_list, IMAGE_DIR, MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR, mask_threshold, padding, confidence_level=0.5):
        self.IDRI_list = LIDC_Patients_list
        self.img_path = IMAGE_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding,padding),(padding,padding),(0,0)]
        self.meta = pd.DataFrame(index=[],columns=['patient_id','original_image','mask_image','malignancy','is_cancer','is_clean'])

    def calculate_malignancy(self,nodule):
        # Calculate the malignancy of a nodule with the annotations made by 4 doctors. Return median high of the annotated cancer, True or False label for cancer
        # if median high is above 3, we return a label True for cancer
        # if it is below 3, we return a label False for non-cancer
        # if it is 3, we return ambiguous
        list_of_malignancy =[]
        for annotation in nodule:
            list_of_malignancy.append(annotation.malignancy)

        malignancy = median_high(list_of_malignancy)
        if  malignancy > 3:
            return malignancy,True
        elif malignancy < 3:
            return malignancy, False
        else:
            return malignancy, 'Ambiguous'
    def save_meta(self,meta_list):
        """Saves the information of nodule to csv file"""
        tmp = pd.Series(meta_list,index=['patient_id','original_image','mask_image','malignancy','is_cancer','is_clean'])
        self.meta = self.meta.append(tmp,ignore_index=True)

    def prepare_dataset(self):
        # This is to name each image and mask
        prefix = [str(x).zfill(3) for x in range(1000)]
        patient_slice_range = np.array([[0,0]])
        total_slice = 0
        # Make directory
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.clean_path_img):
            os.makedirs(self.clean_path_img)
        if not os.path.exists(self.clean_path_mask):
            os.makedirs(self.clean_path_mask)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)


        for patient in tqdm(self.IDRI_list):
            pid = patient #LIDC-IDRI-0001~

            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            nodules_annotation = scan.cluster_annotations()
            vol = scan.to_volume()
            print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid,vol.shape,len(nodules_annotation)))
            nodule_min_start = vol.shape[2]
            nodule_max_stop = -1
            patient_image_dir = IMAGE_DIR + '/'+ pid
            patient_mask_dir = MASK_DIR + '/' + pid
            Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
            Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)
            if len(nodules_annotation) > 0:
                # Patients with nodules
                for nodule_idx, nodule in enumerate(nodules_annotation):
                # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                # This current for loop iterates over total number of nodules in a single patient
                    # mask => the final mask
                    # cbbox => the bounding box of 3d nodule
                    mask, cbbox, masks = consensus(nodule,self.c_level,self.padding)
                    # retrieve the input image slice with nodule
                    lung_np_array = vol[cbbox]
                    nodule_min_start = min(nodule_min_start, cbbox[2].start)
                    nodule_max_stop = max(nodule_max_stop, cbbox[2].stop)
                    # We calculate the malignancy information
                    malignancy, cancer_label = self.calculate_malignancy(nodule)

                    for nodule_slice in range(mask.shape[2]):
                        # This second for loop iterates over each single nodule.
                        # There are some mask sizes that are too small. These may hinder training.
                        if np.sum(mask[:,:,nodule_slice]) <= self.mask_threshold:
                            continue
                        nodule_name = "{}_img_slice{}".format(pid[-4:],prefix[nodule_slice+cbbox[2].start])
                        mask_name = "{}_mask_slice{}".format(pid[-4:],prefix[nodule_slice+cbbox[2].start])
                        if os.path.isfile(patient_image_dir + '/' +  nodule_name + '.npy'):
                            old_mask = np.load(patient_mask_dir + '/' +  mask_name + '.npy')
                            new_mask = old_mask + mask[:,:,nodule_slice]
                            np.save(patient_mask_dir + '/' + mask_name,new_mask)
                        else:
                            total_slice += 1
                            meta_list = [pid[-4:],nodule_name,mask_name,malignancy,cancer_label,False]
                            self.save_meta(meta_list)
                            np.save(patient_image_dir + '/' +  nodule_name,lung_np_array[:,:,nodule_slice])
                            np.save(patient_mask_dir + '/' + mask_name,mask[:,:,nodule_slice]) 
                patient_slice_range = np.append(patient_slice_range, np.array([[nodule_min_start, nodule_max_stop]]), axis=0)
        return patient_slice_range, total_slice

    def prepare_dataset_clean(self, patient_slice_range, total_slice):
            print('start prepare clean dataset ...')
            # This is to name each image and mask
            prefix = [str(x).zfill(3) for x in range(1000)]
            per_nodule_slice = total_slice / len(self.IDRI_list)
            positive_patient_count = 1
            # Make directory
            if not os.path.exists(self.img_path):
                os.makedirs(self.img_path)
            if not os.path.exists(self.mask_path):
                os.makedirs(self.mask_path)
            if not os.path.exists(self.clean_path_img):
                os.makedirs(self.clean_path_img)
            if not os.path.exists(self.clean_path_mask):
                os.makedirs(self.clean_path_mask)
            if not os.path.exists(self.meta_path):
                os.makedirs(self.meta_path)

            for patient in tqdm(self.IDRI_list):
                pid = patient #LIDC-IDRI-0001~

                scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
                nodules_annotation = scan.cluster_annotations()
                vol = scan.to_volume()
                print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid,vol.shape,len(nodules_annotation)))
                patient_clean_dir_image = CLEAN_DIR_IMAGE + '/' + pid
                patient_clean_dir_mask = CLEAN_DIR_MASK + '/' + pid
                Path(patient_clean_dir_image).mkdir(parents=True, exist_ok=True)
                Path(patient_clean_dir_mask).mkdir(parents=True, exist_ok=True)

                if len(nodules_annotation) > 0:
                    [stop_1, start_2] = patient_slice_range[positive_patient_count]
                    # 分前後兩部分
                    if stop_1 > per_nodule_slice/2:
                        start_1 = stop_1 - per_nodule_slice/2
                    else: 
                        start_1 = 0
                    if start_2 + per_nodule_slice/2 <= vol.shape[2]:
                        stop_2 = start_2 + per_nodule_slice/2
                    else: 
                        stop_2 = vol.shape[2]
                    # 前半部分
                    for slice_id in range(int(start_1), int(stop_1)):
                        # if slice >= per_nodule_slice:
                        #     break
                        # lung_segmented_np_array = segment_lung(vol[:,:,slice_id])
                        # lung_segmented_np_array[lung_segmented_np_array==-0] =0
                        lung_mask = np.zeros_like(vol[:,:,slice_id])

                        #CN= CleanNodule, CM = CleanMask
                        nodule_name = "{}_img_slice{}".format(pid[-4:],prefix[slice_id])
                        mask_name = "{}_mask_slice{}".format(pid[-4:],prefix[slice_id])
                        meta_list = [pid[-4:],nodule_name,mask_name,0,False,True]
                        self.save_meta(meta_list)
                        np.save(patient_clean_dir_image + '/' + nodule_name, vol[:,:,slice_id])
                        np.save(patient_clean_dir_mask + '/' + mask_name, lung_mask)
                    # 後半部分    
                    for slice_id in range(int(start_2), int(stop_2)):
                        # if slice >= per_nodule_slice:
                        #     break
                        # lung_segmented_np_array = segment_lung(vol[:,:,slice_id])
                        # lung_segmented_np_array[lung_segmented_np_array==-0] =0
                        lung_mask = np.zeros_like(vol[:,:,slice_id])

                        #CN= CleanNodule, CM = CleanMask
                        nodule_name = "{}_img_slice{}".format(pid[-4:],prefix[slice_id])
                        mask_name = "{}_mask_slice{}".format(pid[-4:],prefix[slice_id])
                        meta_list = [pid[-4:],nodule_name,mask_name,0,False,True]
                        self.save_meta(meta_list)
                        np.save(patient_clean_dir_image + '/' + nodule_name, vol[:,:,slice_id])
                        np.save(patient_clean_dir_mask + '/' + mask_name, lung_mask)
                    positive_patient_count += 1
                   
                else:
                    #There are patients that don't have nodule at all. Meaning, its a clean dataset. We need to use this for validation
                    slice_count = 0
                    if vol.shape[2] >= per_nodule_slice:
                        clean_slice = slice(int((vol.shape[2]-per_nodule_slice)/2), int((vol.shape[2]+per_nodule_slice)/2), 1)
                        slice_count = int((vol.shape[2]-per_nodule_slice)/2)
                    else:
                        clean_slice = slice((0, vol.shape[2]), 1)
                    
                    for slice_vol in vol.transpose(2,0,1)[clean_slice]:
                        # if slice >= per_nodule_slice:
                        #     break
                        # lung_segmented_np_array = segment_lung(slice_vol)
                        # lung_segmented_np_array[lung_segmented_np_array==-0] =0
                        lung_mask = np.zeros_like(slice_vol)

                        #CN= CleanNodule, CM = CleanMask
                        nodule_name = "{}_img_slice{}".format(pid[-4:],prefix[slice_count])
                        mask_name = "{}_mask_slice{}".format(pid[-4:],prefix[slice_count])
                        meta_list = [pid[-4:],nodule_name,mask_name,0,False,True]
                        self.save_meta(meta_list)
                        np.save(patient_clean_dir_image + '/' + nodule_name, slice_vol)
                        np.save(patient_clean_dir_mask + '/' + mask_name, lung_mask)
                        slice_count += 1

            print("Saved Meta data")
            self.meta.to_csv(self.meta_path+'meta_info.csv',index=False)



if __name__ == '__main__':
    # I found out that simply using os.listdir() includes the gitignore file 
    LIDC_IDRI_list= [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()

    test= MakeDataSet(LIDC_IDRI_list,IMAGE_DIR,MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR,mask_threshold,padding,confidence_level)
    patient_slice_range, total_slice = test.prepare_dataset()
    total_slice /= POS_NEG_RATIO
    test.prepare_dataset_clean(patient_slice_range, total_slice)

