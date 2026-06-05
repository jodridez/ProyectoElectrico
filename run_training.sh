#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --job-name="crosssim_training"
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err
#SBATCH --mail-user=CORREO@ucr.ac.cr
#SBATCH --mail-type=BEGIN,END,FAIL
# Activar ambiente
mamba activate crosssim
# Ir al directorio de training
cd /home/$USER/ProyectoElectrico/training
# Verificar GPU disponible
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Ejecutar training
# Parametros en MLP_training.py: Nruns=5, Nepochs=20, batchsize=10
python MLP_training.py