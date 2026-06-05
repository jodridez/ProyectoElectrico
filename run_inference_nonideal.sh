#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --job-name="crosssim_inf_mote2"
#SBATCH --output=inference_mote2_%j.out
#SBATCH --error=inference_mote2_%j.err
#SBATCH --mail-user=CORREO@ucr.ac.cr
#SBATCH --mail-type=BEGIN,END,FAIL
# Antes de enviar, configurar inference/inference_config.py:
#
# Rmin = <valor CELEQ en ohms — estado LRS>
# Rmax = <valor CELEQ en ohms — estado HRS>
# infinite_on_off_ratio = False
#
# error_model = "generic"
# alpha_error = <derivado del analisis estadistico de la LUT MoTe2>
#
# noise_model = "generic"
# alpha_noise = <de mediciones C2C, o 0.00 si no disponible>
#
# drift_model = "none"
# t_drift = 0
#
# Referencia Ion/Ioff ~ 10^3 (Rupom et al. 2025):
# Rmin ~ 1e3 ohm Rmax ~ 1e6 ohm
mamba activate crosssim
cd /home/$USER/ProyectoElectrico/inference
python run_inference.py