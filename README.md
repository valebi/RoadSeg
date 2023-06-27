# Roadseg

## Getting Started

Run the following commands to setup code environment:
```bash
make install
```
This will install the pre-commit hooks and the required packages.

For windows type
 ```bash
conda activate {YOUR_ENV}
& 'make.exe' install
```
## Running on Kaggle
You need to add the following kaggle secrets:
- Wandb API key (goto wandb > user settings)
- Github ssh key (goto github > settings > personal access tokens)
@TODO this file may have copyright issues

## Running on Euler w. Kaggle Datasets
1. Clone the git repo to /cluster/home/{user}/cil 
[ git clone https://TOKEN@github.com/valebi/RoadSeg.git ]
2. Get a kaggle.json authentication file from https://www.kaggle.com/settings > API > generate token and upload it to RoadSeg
3. Run sh download_data.sh
4. Run sh euler.sh
If you get a weird ${}-command-not-found error, run 
'''
sed -i.bak 's/\r$//' euler.sh
'''