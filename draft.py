import SimpleITK as sitk
import numpy as np


img = sitk.ReadImage('/homes/rqyu/Data/PI-RADS/original_data/2012-2016-CA_formal_BSL^bai song lai ^^6698-8/t2_Resize.nii')
img = sitk.GetArrayFromImage(img)
print(img.shape)
