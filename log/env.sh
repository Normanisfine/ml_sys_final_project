# on rtx8000

#gs_env
module purge
module load anaconda3/2024.02
module load cuda/11.6.2

cd /vast/ml8347/ml_sys/final_project/gs_profile

export CONDA_PKGS_DIRS="/vast/ml8347/ml_sys/final_project/envs/.conda/pkgs"
export CONDA_ENVS_DIRS="/vast/ml8347/ml_sys/final_project/envs/.conda/envs"

conda create --prefix /vast/ml8347/ml_sys/final_project/envs/gs_env python=3.7.13 -c pytorch -c conda-forge -c defaults


conda activate /vast/ml8347/ml_sys/final_project/envs/gs_env


conda install plyfile pip=22.3.1 pytorch=1.12.1 torchaudio=0.12.1 torchvision=0.13.1 tqdm -c pytorch -c conda-forge -c defaults


pip install submodules/diff-gaussian-rasterization



pip install submodules/simple-knn
pip install submodules/fused-ssim

pip install opencv-python
pip install joblib


#-------------------------------------tcgs_env-------------------------------------

module purge
module load anaconda3/2024.02
module load cuda/11.6.2

cd /vast/ml8347/ml_sys/final_project/tcgs_profile

export CONDA_PKGS_DIRS="/vast/ml8347/ml_sys/final_project/envs/.conda/pkgs"
export CONDA_ENVS_DIRS="/vast/ml8347/ml_sys/final_project/envs/.conda/envs"

conda create --prefix /vast/ml8347/ml_sys/final_project/envs/tcgs_env python=3.7.13 -c pytorch -c conda-forge -c defaults

conda activate /vast/ml8347/ml_sys/final_project/envs/tcgs_env



conda install plyfile pip=22.3.1 pytorch=1.12.1 torchaudio=0.12.1 torchvision=0.13.1 tqdm -c pytorch -c conda-forge -c defaults


pip install submodules/tcgs_speedy_rasterizer

pip install submodules/simple-knn
pip install submodules/fused-ssim

pip install opencv-python
pip install joblib