import numpy as np
import h5py

def LoadDataSetSingleCoil(load_dir, variable = "data_fs"):
    f = h5py.File(load_dir, 'r')
    if variable == "data_fs" or variable == "us_masks":
        data = np.expand_dims(np.transpose(np.array(f[variable]), (0,2,1)), axis=1).astype(np.float32)
    elif variable == "data_us":
        data = np.transpose(np.array(f[variable]), (1,0,3,2))
        phase_ = (data[:,1,:,:] * 2 * np.pi) - np.pi
        data = data[:,0,:,:] * np.exp(1j * phase_)
    return data

def LoadDataSetMultiCoil(load_dir, variable = 'images_fs', padding = True, Norm = True, res = [384, 384], slices = 10, is_complex = True, channel_cat = False):
    f = h5py.File(load_dir,'r')

    if np.array(f[variable]).ndim==3:
        data=np.expand_dims(np.transpose(np.array(f[variable]),(0,1,2)),axis=1)
    else:
        data=np.transpose(np.array(f[variable]),(1,0,2,3))

    if is_complex:
        data  = data['real'] + 1j*data['imag']    
    else:
        data=data.astype(np.float32)

    if Norm:
        #normalize each subject    
        subjects=int(data.shape[0]/slices )
        data=np.split(data,subjects,axis=0)  
        data=[x/abs(x).max() for x in data]
        data=np.concatenate(data,axis=0)  
    
    if channel_cat:    
        data  = np.concatenate((data.real, data.imag), axis=1)

    if padding:
        pad_x = int((res[0]-data.shape[2])/2); pad_y = int((res[1]-data.shape[3])/2)
        data=np.pad(data,((0,0),(0,0),(pad_x, pad_x),(pad_y, pad_y)))  

    return data

def get_fs_singlecoil(data_dir, phase = 'train'):
    """
    Returns fully sampled singlecoil images of shape
    [Number of images x height x width]
    """

    print("Loading T1 images")
    target_file = data_dir + "T1_2_multi_synth_recon_" + str(phase) + ".mat"
    data_fs_t1 = LoadDataSetSingleCoil(target_file)

    print("Loading T2 images")
    target_file = data_dir + "T2_2_multi_synth_recon_" + str(phase) + ".mat"
    data_fs_t2 = LoadDataSetSingleCoil(target_file)

    print("Loading PD images")
    target_file = data_dir + "PD_2_multi_synth_recon_" + str(phase) + ".mat"
    data_fs_pd = LoadDataSetSingleCoil(target_file)

    data_fs = np.concatenate((data_fs_t1, data_fs_t2, data_fs_pd), axis = 0)
    data_fs = np.squeeze(data_fs)
    print("data_fs.shape: " + str(data_fs.shape))
    return data_fs


def get_us_singlecoil(data_dir, phase = 'test', contrast= 'T1', R = 4):
    """
    Returns singlecoil undersampled images and undersampling masks of shape
    [Number of images x height x width]
    """

    print("Loading " + contrast + " " + str(R) + "x Data")
    target_file = data_dir + contrast + "_" + str(R) + "_multi_synth_recon_" + str(phase) + ".mat"
    data_us = LoadDataSetSingleCoil(target_file, variable = "data_us")
    us_masks = LoadDataSetSingleCoil(target_file, variable = "us_masks")

    data_us = np.squeeze(data_us)
    us_masks = np.squeeze(us_masks)

    print(data_us.shape, us_masks.shape)

    return data_us, us_masks

def get_fs_multicoil(data_dir, phase = 'train'):
    """ Returns fully sampled multicoil images of shape
    [Number of images x height x width]
    """

    print("Loading T1 images")
    target_file = data_dir + "T1/T1_under_sampled_2x_multicoil_" + str(phase)+ ".mat"
    data_fs_t1 = LoadDataSetMultiCoil(target_file)

    print("Loading T2 images")
    target_file = data_dir + "T2/T2_under_sampled_2x_multicoil_" + str(phase) + ".mat"
    data_fs_t2 = LoadDataSetMultiCoil(target_file)

    print("Loading FLAIR images")
    target_file = data_dir + "FLAIR/FLAIR_under_sampled_2x_multicoil_" + str(phase)+ ".mat"
    data_fs_flair=LoadDataSetMultiCoil(target_file)
   
    data_fs=np.concatenate((data_fs_t1,data_fs_t2,data_fs_flair),axis=0)
    data_fs = np.squeeze(data_fs)

    print(data_fs.shape)

    return data_fs

def get_us_multicoil(data_dir, phase='test', contrast= 'T1', R = 4):
    """
    Returns undersampled images of shape
    [Number of images x number of coils x height x width],
    undersampling masks of shape
    [Number of images x 1 x height x width],
    coil sensitivity maps of shape
    [Number of images x number of coils x height x width]
    """

    print("Loading " + contrast + " " + str(R) + "x Data")

    target_file = data_dir + contrast + "/" + contrast + "_under_sampled_" + str(R) + "x_multicoil_" + str(phase) + ".mat"
    data_fs = LoadDataSetMultiCoil(target_file, 'images_fs', padding = False, Norm = True, channel_cat = False)
    data_us = LoadDataSetMultiCoil(target_file, 'images_us', padding = False, Norm = True, channel_cat = False)        
    masks = LoadDataSetMultiCoil(target_file, 'map', padding = False, Norm = False, is_complex = False, channel_cat = False)                    
    coil_maps = LoadDataSetMultiCoil(target_file, 'coil_maps', padding = False, Norm = False, channel_cat = False)

    print(data_us.shape, masks.shape, coil_maps.shape)

    return data_us, masks, coil_maps
