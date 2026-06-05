#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --job-name="crosssim_smoketest"
#SBATCH --output=smoketest_training_%j.out
#SBATCH --error=smoketest_training_%j.err
#SBATCH --mail-user=CORREO@ucr.ac.cr
#SBATCH --mail-type=BEGIN,END,FAIL
# Sustituir CORREO con el correo institucional @ucr.ac.cr
echo "========================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Nodo : $SLURMD_NODENAME"
echo "Inicio : $(date)"
echo "========================================"
mamba activate crosssim
# Verificar GPU
python3 -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
cd /home/$USER/ProyectoElectrico/training
# Nruns=1, Nepochs=1, batchsize=200 configurados en MLP_training.py
python MLP_training.py
echo "========================================"
echo "Fin : $(date)"
echo "========================================"
Enviar a cola y monitorear:
sbatch run_smoketest_training.sh
# Verificar que entro a cola
squeue --me
# Ver output en tiempo real
tail -f smoketest_training_<JOBID>.out
# Ver en que nodo esta corriendo
squeue --me
# Columna NODELIST muestra el nodo, ej: cngpu001
# Monitorear GPU desde ese nodo
ssh -t cngpu001 nvidia-smi