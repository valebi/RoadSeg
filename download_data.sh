
USER="$(whoami)"
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
unzip -q ethz-cil-road-segmentation-2023.zip
cd -
kaggle datasets download valebi/hofmann-osm --unzip --path roadseg-download-openstreetmap
kaggle datasets download valebi/google-roadseg --unzip --path google-roadseg
#kaggle kernels output valgitebi/roadseg-download-openstreetmap --path roadseg-download-openstreetmap
kaggle datasets download valebi/esri-streetmap-tiles --unzip --path esri-streetmap-tiles
kaggle datasets download selinnbaris/processed-bing-dataset --unzip --path processed-bing-dataset
#kaggle kernels output ahmetalperozudogru/bingscrape-noarrow --path bingscrape-noarrow
#kaggle datasets  download esri-streetmap-tiles --unzip --path esri-streetmap-tiles
wget -O roadtracing.zip "https://polybox.ethz.ch/index.php/s/USLJotE9cgtZPMr/download?path=%2F&files=roadtracing.zip"
#unzip roadtracing.zip -q
#rm roadtracing.zip
zip -FFv roadtracing.zip --out roadtracing_fixed.zip
unzip -q roadtracing_fixed.zip

wget -O epfl-roadseg.zip "https://polybox.ethz.ch/index.php/s/USLJotE9cgtZPMr/download?path=%2F&files=epfl.zip"
unzip -q epfl-roadseg.zip
rm epfl-roadseg.zip

#wget -O deepglobe.zip "https://polybox.ethz.ch/index.php/s/USLJotE9cgtZPMr/download?path=%2F&files=deepglobe.zip"
#unzip deepglobe.zip -q
#rm deepglobe.zip

