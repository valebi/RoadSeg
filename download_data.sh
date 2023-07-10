
USER="bieriv"
DATA_DIR="roadseg"

module load eth_proxy
 # setup kaggle
mkdir /cluster/home/${USER}/.kaggle -p
cp kaggle.json /cluster/home/${USER}/.kaggle/
chmod 600 /cluster/home/${USER}/.kaggle/kaggle.json
pip install -q kaggle
kaggle config view
#rm kaggle.json
# download data
cd  /cluster/scratch/${USER}
mkdir ${DATA_DIR} -p
cd ${DATA_DIR}
kaggle datasets download valebi/maptiler-custom-tiles --unzip --path maptiler-custom-tiles
kaggle competitions download ethz-cil-road-segmentation-2023 --path ethz-cil-road-segmentation-2023 --force
cd ethz-cil-road-segmentation-2023
unzip ethz-cil-road-segmentation-2023.zip -q
cd -
kaggle datasets download valebi/hofmann-osm --unzip --path roadseg-download-openstreetmap
#kaggle kernels output valebi/roadseg-download-openstreetmap --path roadseg-download-openstreetmap
kaggle datasets download valebi/esri-streetmap-tiles --unzip --path esri-streetmap-tiles
kaggle datasets download selinnbaris/processed-bing-dataset --unzip --path processed-bing-dataset
#kaggle kernels output ahmetalperozudogru/bingscrape-noarrow --path bingscrape-noarrow
#kaggle datasets  download esri-streetmap-tiles --unzip --path esri-streetmap-tiles
