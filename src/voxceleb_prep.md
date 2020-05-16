
to do list:

- add kaldi commit number 
- add inference script
- add data prep document

### Training/Test data: VOXCELEB 1 and VOXCELEB 2 data.
 
 #### 1. Download Voxceleb1 and Voxceleb2 from [this website](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/). 
 - You need to provide your affiliation info and email address by filling out [this form](https://docs.google.com/forms/d/e/1FAIpQLSdQhpq2Be2CktaPhuadUMU7ZDJoQuRlFlzNO45xO-drWQ0AXA/viewform?fbzx=7440236747203254000) to get the ID/PW. DownloadaAudio files only.
 
 - Check if you downloaded the following files. In total, Voxceleb1 is ~33G and Voxceleb2 is ~74G. 
 
```
vox1_dev_wav_partaa.zip
vox1_dev_wav_partab
vox1_dev_wav_partac
vox1_dev_wav_partad
vox1_test_wav.zip
vox2_dev_aac_partaa.zip
vox2_dev_aac_partab
vox2_dev_aac_partac
vox2_dev_aac_partad
vox2_dev_aac_partae
vox2_dev_aac_partaf
vox2_dev_aac_partag
vox2_dev_aac_partah
vox2_test_aac.zip 
```
- Use the following commands to create a concatenated zip file.

```
 $ cat vox1_dev* > vox1_dev_wav.zip
 $ cat vox2_dev_mp4* > vox2_mp4.zip
```

#### 2. Unzip the following 4 files.
```
vox1_dev_wav.zip
vox1_test_wav.zip
vox2_aac.zip
vox2_test_aac.zip 
```
- from vox1, you get *wav* folders. From vox2, you get *aac* and *dev/aac* folders.
  
  
vox1_dev_wav.zip > ./wav  
vox1_test_wav.zip > ./wav  
vox2_aac.zip > ./aac  
vox2_test_aac.zip  > ./dev/aac   



#### 3. Arrange the folders as the folder structure shown below.
```
./path/to/vox1
├── dev
│   └── wav
└── test
    └── wav
./path/to/vox2
├── dev
│   └── aac
└── test
    └── aac
```

#### 4. *vox1* and *vox2* will be the dataset folders. Plug in these paths to the script files.
 
### Dataset for augmentation

#### 1. Download RIRS (Impulse responses) from openslr.org

- Download RIRS from [this link](http://www.openslr.org/resources/28/rirs_noises.zip)

#### 1. Unzip the zip file in the base folder

```
$ unzip rirs_noises.zip
```
This will create *RIRS_NOISES* folder in your base directory and it will be recognized by the script.


