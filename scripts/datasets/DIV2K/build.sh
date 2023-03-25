mkdir datasets
mkdir datasets/DIV2K

# download DIV2K
cd datasets/DIV2K
## HR
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
## X2
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip
## X3
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip
## X4
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip

# unzip 
unzip DIV2K_train_HR.zip
unzip DIV2K_train_LR_bicubic_X2.zip
unzip DIV2K_train_LR_bicubic_X3.zip
unzip DIV2K_train_LR_bicubic_X4.zip
unzip DIV2K_valid_HR.zip
unzip DIV2K_valid_LR_bicubic_X2.zip
unzip DIV2K_valid_LR_bicubic_X3.zip
unzip DIV2K_valid_LR_bicubic_X4.zip
cd ../..

# move
mkdir datasets/DIV2K/train
mkdir datasets/DIV2K/train/HR/
mkdir datasets/DIV2K/train/LR/
mv datasets/DIV2K/DIV2K_train_HR datasets/DIV2K/train/HR/original/
mkdir datasets/DIV2K/train/LR/bicubic/
mkdir datasets/DIV2K/train/LR/bicubic/X2/
mv datasets/DIV2K/DIV2K_train_LR_bicubic/X2 datasets/DIV2K/train/LR/bicubic/X2/original
mkdir datasets/DIV2K/train/LR/bicubic/X3/
mv datasets/DIV2K/DIV2K_train_LR_bicubic/X3 datasets/DIV2K/train/LR/bicubic/X3/original
mkdir datasets/DIV2K/train/LR/bicubic/X4/
mv datasets/DIV2K/DIV2K_train_LR_bicubic/X4 datasets/DIV2K/train/LR/bicubic/X4/original
rm datasets/DIV2K/DIV2K_train_LR_bicubic -r
mkdir datasets/DIV2K/valid
mkdir datasets/DIV2K/valid/HR/
mkdir datasets/DIV2K/valid/LR/
mv datasets/DIV2K/DIV2K_valid_HR datasets/DIV2K/valid/HR/original
mkdir datasets/DIV2K/valid/LR/bicubic/
mkdir datasets/DIV2K/valid/LR/bicubic/X2/
mv datasets/DIV2K/DIV2K_valid_LR_bicubic/X2 datasets/DIV2K/valid/LR/bicubic/X2/original
mkdir datasets/DIV2K/valid/LR/bicubic/X3/
mv datasets/DIV2K/DIV2K_valid_LR_bicubic/X3 datasets/DIV2K/valid/LR/bicubic/X3/original
mkdir datasets/DIV2K/valid/LR/bicubic/X4/
mv datasets/DIV2K/DIV2K_valid_LR_bicubic/X4 datasets/DIV2K/valid/LR/bicubic/X4/original
rm datasets/DIV2K/DIV2K_valid_LR_bicubic -r

# extract_subimages
python scripts/datasets/DIV2K/extract_subimages.py

# # create lmdb
python scripts/datasets/DIV2K/create_lmdb.py

# # generate meta info
python scripts/datasets/DIV2K/generate_meta_info.py
