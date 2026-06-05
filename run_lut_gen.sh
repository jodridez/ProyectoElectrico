#!/bin/bash -l
#SBATCH --partition=serial
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --job-name="lut_gen_mote2"
#SBATCH --output=lut_gen_%j.out
#SBATCH --error=lut_gen_%j.err
#SBATCH --mail-user=CORREO@ucr.ac.cr
#SBATCH --mail-type=END,FAIL
# Antes de enviar:
# 1. Subir set_MoTe2.csv y reset_MoTe2.csv al cluster:
# scp set_MoTe2.csv reset_MoTe2.csv USER@172.16.24.2:/home/USER/
# ProyectoElectrico/examples/lookup_table_generation/
#
# 2. Editar create_lookup_table.py:
# folder = 'MoTe2_LUT'
# set_file = 'set_MoTe2.csv'
# reset_file = 'reset_MoTe2.csv'
# read_voltage = <Vread del equipo CELEQ en V>
# (si datos en conductancia S, usar read_voltage = 1)
mamba activate crosssim
cd /home/$USER/ProyectoElectrico/examples/lookup_table_generation
# Verificar que los CSV existen
[ ! -f "set_MoTe2.csv" ] && echo "ERROR: set_MoTe2.csv no encontrado" && exit 1
[ ! -f "reset_MoTe2.csv" ] && echo "ERROR: reset_MoTe2.csv no encontrado" && exit 1
python create_lookup_table.py
# Verificar salida
ls -lh MoTe2_LUT/
# Debe mostrar dG_increasing.txt y dG_decreasing.txt