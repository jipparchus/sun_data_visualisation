# About this repository
The main routine aims to demonstrate an example of visualising sun observation data. The observation data were obtained from the Atmospheric Imaging Assembly on board the Solar Dynamic Observatory ([AIA/SDO](https://sdo.gsfc.nasa.gov/mission/instruments.php)) and downloaded from [the Joint Science Operations Center (JSOC)](http://jsoc.stanford.edu/).

SDO is a space mission launched in 2010 that observes the Sun 24/7. AIA/SDO is one of the instruments installed on SDO, and its observation is in continuous light and in the extreme ultraviolet (EUV) range.

The data used here consist of Extreme Ultraviolet (EUV) observations at 171.1 nm, 19.3 nm, and 30.4 nm, mainly corresponding to plasma emissions at temperatures of 0.6 MK, 1.2 MK & 20 MK, and 0.05 MK, respectively. There are 26 observation data for each channel observed at different times from the 19th of July 2012, 07:30 (UT) to 07:45 (UT) of the same day with approximately 30-second intervals. The field of view and the observation date were selected so that the spatial and temporal window capture an excellent example of off-limb flares and a series of 'coronal rains' as flows of plasma blobs.

<video src="aia_171.mp4" controls="true" width="400"></video>
<video src="aia_193.mp4" controls="true" width="400"></video>
<video src="aia_304.mp4" controls="true" width="400"></video>

(videos were created through the [link](https://sdo.gsfc.nasa.gov/data/aiahmi/browse/queued.php))

# The main routine will ...

- Download the time-series observation data from this github repository.

- Enhances the coronal loop structures visibility by using Gaussian filter, and create time-series animations.

- Cuts out images at each time step at each channel (17.1 nm, 19.3 nm, and 30.4 nm) along a distinctive loop found  at 17.1 nm image. For each channel, the cut out images are averaged in axis across the path and are aligned along time axis to produce a time-distance plot. The coordinates of the path are obtained by spline interpolation of several points selected along the loop.

- Estimating a typical speed of plasma flows by approximating a distance of the cutout path as a line of segment connecting the starting and the ending points.


# Anaconda environment prep

- Please set up the environment to run ``` main.py ``` by executing the below. The name of your environment, "foo", can be set to anything you like.

```bash
conda env create -n foo -f requirement.yml
```
```bash
conda activate foo
```

# Run the main code
- Please download ``` main.py ``` and place it in an arbitrary directory.

- Move to the directory and try:
    ```bash
    python main.py
    ```

- ``` main.py ``` will automatically download necessary dataset, ``` /aia_data ```, to the same directory as ``` main.py ```. If the download fails, please download ``` /aia_data ``` manually.

# Others

- ``` download_data.py ``` is not called in ``` main.py ```. This can be used when you want to download the AIA/SDO observation data from  JSOC [Data Export Request page](http://jsoc.stanford.edu/ajax/exportdata.html?ds=aia.lev1_euv_12s). You may need to register your **email address** both in JSOC [email registration page](http://jsoc.stanford.edu/ajax/register_email_art.html) and in ``` download_data.py ``` by editing the second argument of the function ``` download_dataset_from_jsoc() ```.