#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --job-name="crosssim_inf_ideal"
#SBATCH --output=inference_ideal_%j.out
#SBATCH --error=inference_ideal_%j.err
#SBATCH --mail-user=CORREO@ucr.ac.cr
#SBATCH --mail-type=BEGIN,END,FAIL
# Verificar antes de enviar que inference_config.py tenga:
# alpha_error = 0.00
# alpha_noise = 0.00
# infinite_on_off_ratio = True
mamba activate crosssim
cd /home/$USER/ProyectoElectrico/inference
python run_inference.py