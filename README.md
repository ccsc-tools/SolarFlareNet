## Operational prediction of solar flares using a transformer-based framework<br>
[![DOI](https://github.com/ccsc-tools/zenodo_icons/blob/main/icons/solarflarenet.svg)](https://zenodo.org/doi/10.5281/zenodo.10080716)


## Authors
Yasser Abduallah, Jason T. L. Wang, Haimin Wang, and Yan Xu

## Abstract

Solar flares are explosions on the Sun. They happen when energy stored in magnetic fields around solar active regions (ARs) is suddenly released. Solar flares and accompanied coronal mass ejections are sources of space weather, which negatively affects a variety of technologies at or near Earth, ranging from blocking high-frequency radio waves used for radio communication to degrading power grid operations. Monitoring and providing early and accurate prediction of solar flares is therefore crucial for preparedness and disaster risk management. In this article, we present a transformer-based framework, named SolarFlareNet, for predicting whether an AR would produce a y-class flare within the next 24 to 72 h. We consider three classes, namely the M5.0 class, the M class and the C class, and build three transformers separately, each corresponding to a class. Each transformer is used to make predictions of its corresponding y-class flares. The crux of our approach is to model data samples in an AR as time series and to use transformers to capture the temporal dynamics of the data samples. Each data sample consists of magnetic parameters taken from Space-weather HMI Active Region Patches (SHARP) and related data products. We survey flare events that occurred from May 2010 to December 2022 using the Geostationary Operational Environmental Satellite X-ray flare catalogs provided by the National Centers for Environmental Information (NCEI), and build a database of flares with identified ARs in the NCEI flare catalogs. This flare database is used to construct labels of the data samples suitable for machine learning. We further extend the deterministic approach to a calibration-based probabilistic forecasting method. The SolarFlareNet system is fully operational and is capable of making near real-time predictions of solar flares on the Web.

## Binder

This notebook is Binder enabled and can be run on [mybinder.org](https://mybinder.org/) by using the link below.


### ccsc_solarflare.ipynb (Jupyter Notebook for SolarFlareNet)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ccsc-tools/SolarFlareNet/HEAD?labpath=ccsc_solarflare.ipynb)

Please note that starting Binder might take some time to create and start the image.

Please also note that the execution time in Binder varies based on the availability of resources. The average time to run the notebook is 15-20 minutes, but it could be more.

For the latest updates of the tool refer to https://github.com/deepsuncode/SolareFlareNet

## Installation on local machine

|Library | Version   | Description  |
|---|---|---|
|keras| 2.6.0 | Deep learning API|
|numpy| 1.21.6| Array manipulation|
|scikit-learn| 1.0.1| Machine learning|
|sklearn| latest| Tools for predictive data analysis|
| pandas|p1.5.1| Data loading and manipulation|
| tensorboard| 2.8.0 | Provides the visualization and tooling needed for machine learning|
| tensorflow-gpu| 2.8.0| Deep learning tool for high performance computation |
