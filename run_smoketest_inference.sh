#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --job-name="crosssim_inf_smoketest"
#SBATCH --output=smoketest_inference_%j.out
#SBATCH --error=smoketest_inference_%j.err
#SBATCH --mail-user=CORREO@ucr.ac.cr
#SBATCH --mail-type=BEGIN,END,FAIL
# Verificar antes de enviar que inference_config.py tenga:
# alpha_error = 0.00
# alpha_noise = 0.00
# infinite_on_off_ratio = True
echo "========================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Nodo : $SLURMD_NODENAME"
echo "Inicio : $(date)"
echo "========================================"
mamba activate crosssim
cd /home/$USER/ProyectoElectrico/inference
python run_inference.py
echo "========================================"
echo "Fin : $(date)"
echo "========================================"
sbatch run_smoketest_inference.sh
squeue --me
tail -f smoketest_inference_<JOBID>.out