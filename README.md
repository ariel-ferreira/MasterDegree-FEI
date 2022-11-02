# Master degree work of Ariel Ferreira.
# Electrical Engineering Department of University Center of FEI - São Bernardo do Campo, SP - Brazil
# 2020 - 2022
# Age estimation based on human speech using a 1D convolutional neural network.

# Targets:

estimação aleatória para 7 classes: (1/7)*100% = 14.3%

projeto Ivandro (parte 1):

precisão global para 7 classes: 56.8% (5138 arquivos utilizados no teste - retirados da lista de teste)
precisão global para 4* classes: 58.6% (5138 arquivos utilizados no teste - retirados da lista de teste)

projeto Ivandro (parte 2):

precisão global para 4* classes: 49.2% (20492 arquivos utilizados no teste)
precisão global para 3** classes: 87.8% (20492 arquivos utilizados no teste)

* crianças, jovens, adultos, idosos - 4 grandes classes: matriz, precision, recall, F1-score.
** crianças, mulheres, homems. - 3 grandes classes: matriz, precision, recall, F1-score.

---------------------------

# Agender Dataset:

# dataset original:

20549 arquivos (utterances) para teste 
32527 arquivos (utterances) para treino

# métricas:

total time (min): 1405.891666666668
total time (hrs): 23.431527777777802
total size (MB): 1288.5318511956002
total size (GB): 1.2578621720279999

# dataset normalizado: 

20549 arquivos (utterances) para teste 
32527 arquivos (utterances) para treino

# métricas:

total time (min): 654.0857333332484
total time (hrs): 10.901428888887475
total size (MB): 600.2144253204001
total size (GB): 0.585928101052

# Ambiente da FEI (acesso via VPN):

lpi@10.31.0.102
pwd: nicutron2022

# Ambiente do Rios:

ariel@rodrigosrios.com -p 3722

# Nvidia procedure:

sudo dnf upgrade --refresh -y
sudo dnf install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf update --refresh

dnf install dkms libstdc++.i686 kernel-devel

# Create file /etc/modprobe.d/blacklist-nouveau.conf and add to it:

blacklist nouveau
options nouveau modeset=0

sudo dracut --force
sudo reboot

sudo systemctl stop gdm

# Download the proper run file script:

chmod +x NVIDIA-Linux-x86_64-384.69.run
sudo ./NVIDIA-Linux-x86_64-384.69.run --dkms -s

# To check install:

nvidia-smi

# Intalling CUDA:

wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
chmod +x cuda_11.8.0_520.61.05_linux.run
./cuda_11.8.0_520.61.05_linux.run --extract=$HOME
sudo ./cuda_11.8.0_520.61.05_linux.run

sudo bash -c "echo /usr/local/cuda/lib64/ > /etc/ld.so.conf.d/cuda.conf"
sudo ldconfig

# To check installation:

nvcc --version

# Installing cuDNN:

# Download the proper tar file from https://developer.nvidia.com/rdp/cudnn-download

tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# Download the proper repo file from https://developer.nvidia.com/rdp/cudnn-download

sudo rpm -i cudnn-local-repo-rhel8-8.6.0.163-1.0-1.x86_64.rpm
sudo yum clean all
sudo yum install libcudnn8-8.6.0.163-1.cuda11.8
sudo yum install libcudnn8-devel-8.6.0.163-1.cuda11.8
sudo yum install libcudnn8-samples-8.6.0.163-1.cuda11.8

# Another option:

sudo yum config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora36/x86_64/cuda-fedora36.repo
sudo yum clean all

sudo yum install libcudnn8=8.6.0.163-1.cuda11.8
sudo yum install libcudnn8-devel=8.6.0.163-1.cuda11.8

# Verify the install:

cp -r /usr/src/cudnn_samples_v8/ $HOME

# Go to the writable path.

cd  $HOME/cudnn_samples_v8/mnistCUDNN

# Compile the mnistCUDNN sample.

make clean && make

# Run the mnistCUDNN sample

./mnistCUDNN

# If cuDNN is properly installed and running on your Linux system, you will see a message similar to the following:

Test passed!

# Referências:

https://paperswithcode.com/
https://pyquestions.com/feeding-npy-numpy-files-into-tensorflow-data-pipeline

-----------------------------------------------------------------------------------

## Organizar:

/home/ferreiraa/dataSet/audio/TIMIT/original/archive/timit/
/home/ferreiraa/dataSet/audio/TIMIT/normallized/

dataset preparation:
 
$ python TIMIT_preparation.py /home/ferreiraa/dataSet/audio/TIMIT/original/archive/timit /home/ferreiraa/dataSet/audio/TIMIT/normallized data_lists/TIMIT_all.scp

$ python TIMIT_preparation.py /home/ferreiraa/dataSet/audio/TIMIT/original/archive/timit /home/ferreiraa/dataSet/audio/TIMIT/torch-normalized data_lists/TIMIT_all.scp
 
# Com 6 classes, fazendo a validação carregando o modelo treinado anteriormente:

Validação = 0.1 da base de treino

Total params: 1,712,470
Trainable params: 1,712,470
Non-trainable params: 0

[1.2998859882354736, 0.6088193655014038]
['loss', 'accuracy']
 
# Com 7 classes, fazendo a validação treinando o modelo:
 
Validação = 0.1 da base de treino
 
Total params: 1,712,728
Trainable params: 1,712,728
Non-trainable params: 0
 
[1.116835117340088, 0.5538130402565002]
['loss', 'accuracy']
Finished in 6683.54s] = 1h50min
