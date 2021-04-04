mkdir -p voc_datasets && cd voc_datasets
# download from pjreddie's mirror   
wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar

tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtrainval_11-May-2012.tar