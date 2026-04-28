#
# Copyright 2017 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#


# =================================================================
# ARCHIVO: MLP_training.py (Versión Comentada para Principiantes)
# =================================================================

# Importamos librerías básicas
import numpy as np  # Para hacer cálculos matemáticos (matrices)
import sys, os      # Para manejar carpetas y archivos en el sistema
# Importamos las herramientas específicas de CrossSim (el simulador)
from training_net import train_neural_net    # El motor que entrena la red
from lr_schedule import adjust_learning_rate # La regla que ajusta la velocidad de aprendizaje
import matplotlib.pyplot as plt             # Para crear gráficas al final

# Configura cómo se ven los números en la consola (máximo 4 decimales)
np.set_printoptions(precision=4, suppress=True)


'''
Este script configura simulaciones de entrenamiento para una red neuronal.
Prueba tres casos: 
1. Matemáticas perfectas (Numeric).
2. Un chip real imperfecto (Single LUT).
3. Un conjunto de chips reales con variaciones entre ellos (Multi LUT).
'''


'''
Analog in-memory training setup script (multi-layer perceptron)
========

This file contains the configuration for a set of training simulations
Once parameters are set, run this file directly: python MLP_training.py

By default, this script runs:
	- one set of runs with ideal numeric weight updates (ideal devices)
	- one set of runs using the set specified by "lookup_table_single", which
		contains the LUT of a single device
	- one set of runs using the set specified by "lookup_table_multi", which
		contains LUTs measured on multiple devices. Array elements are randomly
		assigned to these LUTs at the beginning of training to model the effects
		of device-to-device variations in the update LUT

This script can be adapted to run any sequence of simulations desired, or to sweep
parameters (e.g. learning rate)

Before running this script, make sure the following empty sub-directories exist in the
training directory:
- console_output
- cross_sim_models
- sweep_results
- weight_update_stats
'''


# --- CONFIGURACIÓN DE LA TARJETA DE VIDEO (GPU) ---
useGPU = True # Indica si usar la tarjeta de video para ir más rápido (está apagado)
if useGPU:
	gpu_num = 0 # ID de la tarjeta de video a usar
	os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num) # Configura el sistema para usar esa GPU

# Carpeta donde se guardarán los mensajes de texto que salen en la consola
outdir = "console_output/"

### --- PARÁMETROS DEL EXPERIMENTO (SWEEP) ---
saveSweep = True # Si es True, guarda los resultados de precisión en un archivo .csv >> directorio sweep results
Nruns = 1        # Cuántas veces repetiremos todo el experimento desde cero
Nepochs = 1      # Cuántas veces la red leerá el "libro de ejercicios" 1 vez por completo

### --- SELECCIÓN DEL DISPOSITIVO ELECTRÓNICO (Hardware) ---
# Aquí eliges qué tipo de componente electrónico quieres simular
device_type = "DWMTJ" # Por defecto está configurado un dispositivo magnético (DWMTJ)

if device_type == "TaOx":
	## Dispositivos de Óxido de Tantalio (física de Sandia 2016)
	lookup_table_single = "TaOx"
	# Diferentes configuraciones de calidad para estos dispositivos
	lookup_table_multi = "TaOx_medium_set"

elif device_type == "ENODe":
	## Dispositivos RAM electroquímicos
	useGPU = False # Estos no funcionan con GPU por su complejidad
	lookup_table_single = "ENODe"
	lookup_table_multi = "ENODe_multi"

elif device_type == "DWMTJ":
	## Dispositivos de "unión de túnel magnético" (física de espines)
	mechanism = "STT" # El mecanismo físico de movimiento
	temperature = 300 # Temperatura de operación (300 Kelvin es temp. ambiente)
	lookup_table_single = "DWMTJ_"+mechanism+"_"+str(temperature)+"K"
	lookup_table_multi = "DWMTJ_"+mechanism+"_"+str(temperature)+"K_multi"

# --- TAMAÑO DEL LOTE (BATCH SIZE) ---
# Indica cuántas imágenes ve la red antes de corregir sus errores
# 10 significa que ve 10 imágenes y luego ajusta sus neuronas
batchsize = 10

### --- SELECCIÓN DEL EXAMEN (DATASET) ---
# Aquí eliges qué debe aprender la red. El seleccionado es 'mnist' (números escritos a mano)
# task = "iris"
#task = "cyber"
#task = "SPECTF"
# task = "small"
task = "mnist"
# task = "fashion"
# task = "UCI_HAR"

# --- TOPOLOGÍA O "FORMA" DEL CEREBRO (MLP) ---
# Define cuántas neuronas hay en cada capa según la tarea elegida

# MLP network topology
# List of feature map dimensions (x, y, channel), starting from first layer input to last layer output
# Only MLPs are currently supported, so first (x,y) are always (1,1)
# Do not include the bias unit input
# If sizes not supplied to train(), a default size will be chosen (see training_net.py)

if task == "iris":
	sizes = (4,8,3) # Para flores: 4 datos de entrada, 8 neuronas medio, 3 tipos de flor
elif task == "cyber":
	sizes = (256,150,9)
elif task == "SPECTF":
	sizes = (44,22,2)
elif task == "small":
	sizes = (64,36,10)
elif task == "mnist":
	sizes = (784,300,10) # 784 píxeles de entrada, 300 neuronas centro de procesamiento, 10 posibles números (0-9)
elif task == "fashion":
	sizes = (784,300,10)
elif task == "UCI_HAR":
	sizes = (561,200,6)

# --- TRUNCAMIENTO DEL DATASET ---
ntset = 0 # Si quieres usar solo unas pocas imágenes para entrenar (0 = usar todas)
ncset = 0 # Si quieres usar solo unas pocas imágenes para evaluar (0 = usar todas)

# --- GUARDADO Y CARGA DE MODELOS ---
loadCrossSimModels = False # ¿Cargar una red que ya sepa la respuesta? (Falso, empezamos de cero)
saveCrossSimModels = True  # ¿Guardar la red al terminar para no perder el progreso? >> La guarda en archivos .npz en cross sim models/.

# Configuración de rutas para guardar los archivos .npz (donde se guarda el cerebro de la red)
#Como loadCrossSimModels es False, el programa ignora los archivos .npz que ya existen en tu carpeta. Crea pesos nuevos al azar (una red "virgen") cada vez que le das a ejecutar.
#Como saveCrossSimModels es True, al terminar el entrenamiento, el programa escribe los nuevos pesos en el archivo .npz, sobrescribiendo (borrando) lo que había antes.
types = ["numeric","lookup_standard","lookup_multi"]
loadModelPaths = [[None for j in range(3)] for i_run in range(Nruns)]
saveModelPaths = [[None for j in range(3)] for i_run in range(Nruns)]
if saveCrossSimModels and not os.path.isdir('./cross_sim_models/'):
    os.makedirs('./cross_sim_models') # Si la carpeta no existe, el programa la crea


# Rellena la lista de rutas para guardar cada uno de los 3 experimentos. Aqui es donde se define el formato de nombre de los archivos .npz 
for j in range(3):
	for i_run in range(Nruns):
		if loadCrossSimModels:
			loadModelPaths[i_run][j] = "./cross_sim_models/MLP_"+task+"_"+types[j]+"_run"+str(i_run)+".npz"
		if saveCrossSimModels:
			saveModelPaths[i_run][j] = "./cross_sim_models/MLP_"+task+"_"+types[j]+"_run"+str(i_run)+".npz"

# --- DETALLES TÉCNICOS DEL CHIP ---
wtmodel = "BALANCED" # Cómo se representan los números positivos y negativos en el hardware
periodic_carry = False # Una técnica avanzada para mayor precisión (apagada)
pc_Nslices = 2 # number of bit slices, e.g. devices per weight to encode magnitude
#Aunque el acarreo periódico esté apagado arriba, este parámetro define la arquitectura base para cuando se necesite mayor precisión. Indica que se utilizarían 2 "rebanadas" (slices) de bits (es decir, dispositivos individuales) para codificar la magnitud de un solo número. Piensa en esto como tener un componente para el "ajuste grueso" y otro para el "ajuste fino".
pc_number_base = 16 # multiplicative factor between period carry slices
learnbias = (True, True) # Indica si la red puede tener "prejuicios" (sesgos) para aprender mejor
#En las ecuaciones de redes neuronales, el sesgo (bias) es un término independiente (la $b$ en la fórmula de la recta $y = Wx + b$) que ayuda a la red a no estar forzada a pasar por el origen (cero), permitiendo un aprendizaje mucho más flexible.

### --- ACTIVACIÓN (La forma en que las neuronas "disparan") ---
activate = "SIGMOID"       # Función matemática para las capas internas
activate_output = "SOFTMAX" # Función para la capa final (da probabilidades de 0% a 100%)
a2dmodel="NONE"            # Simulación de convertidores analógico-digitales (ninguno)
stochastic_updates = False # Si los cambios en el chip son al azar (Falso)

# --- VELOCIDAD DE APRENDIZAJE (LEARNING RATE) ---
lr_sched = True # ¿Bajar la velocidad de aprendizaje conforme la red avanza? (Sí, ayuda a estabilizar)

# Valores de velocidad de aprendizaje óptimos para cada tarea
# NOTE: The default learning rates below are not necessarily optimal for the chosen
# 	dataset and topology!
if task == "iris":
	alpha_numeric = 0.1
	alpha_lut_standard = 0.02
	alpha_lut_multi = 0.02
elif task == "SPECF":
	alpha_numeric = 0.1
	alpha_lut_standard = 0.1
	alpha_lut_standard = 0.1
elif task == "small":
	alpha_numeric = 0.05
	alpha_lut_standard = 0.006
	alpha_lut_multi = 0.012
elif task == "cyber":
	alpha_numeric = 0.00005
	alpha_lut_standard = 0.000025
	alpha_lut_multi = 0.000025
elif task == "mnist":
	alpha_numeric = 0.01      # Velocidad para el caso ideal
	alpha_lut_standard = 0.0025 # Velocidad para el chip imperfecto
	alpha_lut_multi = 0.001    # Velocidad para muchos chips
elif task == "fashion":
	alpha_numeric = 0.0002
	alpha_lut_standard = 0.0002
	alpha_lut_multi = 0.0002
elif task == "UCI_HAR":
	alpha_numeric = 0.01
	alpha_lut_standard = 0.01
	alpha_lut_multi = 0.01
else:
	raise ValueError("Invalid task") # Error si escribes mal el nombre de la tarea

# --- ESTADÍSTICAS DE ACTUALIZACIÓN ---
# ¿Queremos guardar datos sobre cómo cambian los pesos de las neuronas?
collect_weight_updates = True
if collect_weight_updates:
    # Number of updates to collect statistics over (first layer only)
	# Make sure this doesn't exceed the total number of device updates during training
	# (max # updates = # weights in 1st layer x # training examples x # epochs
	Nupdates_total = 100000 # Cuántas actualizaciones queremos observar
	diagnosticParams = [True,Nupdates_total]
else:
	diagnosticParams = [False,100]

# Contenedores vacíos para guardar los resultados finales
results = np.zeros((Nepochs,3,Nruns))
deltaW_info = [[None for i in range(3)] for j in range(Nruns)]
epoch_vec = np.arange(Nepochs)

# =================================================================
# INICIO DEL BUCLE DE ENTRENAMIENTO (Aquí es donde ocurre la magia)
# =================================================================
print('Training on dataset: '+task+'\n')

for i_run in range(Nruns):

	# Inicializamos el objeto de entrenamiento
	train_net = train_neural_net(outdir)

	# Configuramos los 3 escenarios con sus parámetros específicos
	# Escenario 1: Ideal (Numeric)
	params_numeric = train_net.set_params(task=task,lookup_table=None,a2dmodel=a2dmodel,\
		stochastic_updates=stochastic_updates,wtmodel=wtmodel,learnbias=learnbias,diagnosticParams=diagnosticParams,\
		useGPU=useGPU)

	# Escenario 2: Un chip real (Standard)
	params_standard = train_net.set_params(task=task,lookup_table=lookup_table_single,a2dmodel=a2dmodel,\
		stochastic_updates=stochastic_updates,wtmodel=wtmodel,learnbias=learnbias,diagnosticParams=diagnosticParams,\
		useGPU=useGPU,periodic_carry=periodic_carry,pc_number_base=pc_number_base,pc_Nslices=pc_Nslices)

	# Escenario 3: Muchos chips con variaciones (Multi)
	params_multi = train_net.set_params(task=task,lookup_table=lookup_table_multi,a2dmodel=a2dmodel,\
		stochastic_updates=stochastic_updates,wtmodel=wtmodel,learnbias=learnbias,diagnosticParams=diagnosticParams,\
		useGPU=useGPU,periodic_carry=periodic_carry,pc_number_base=pc_number_base,pc_Nslices=pc_Nslices)

	# --- EJECUCIÓN DE LOS ENTRENAMIENTOS ---

	print('##########')
	print('Numeric (ideal), Run '+str(i_run))
	print('##########')
	# Se llama a la función .train para el caso IDEAL
	results[:,0,i_run], deltaW_info[i_run][0] = train_net.train(filename=task+"_numeric.txt",dataset=task,params=params_numeric,n_epochs=Nepochs,\
		activate=activate,alpha=alpha_numeric,activate_output=activate_output,learnbias=learnbias,ntset=ntset,ncset=ncset,lr_sched=lr_sched,
		lr_sched_function=adjust_learning_rate,batchsize=batchsize,loadModelPath=loadModelPaths[i_run][0],saveModelPath=saveModelPaths[i_run][0],sizes=sizes)

	print('##########')
	print('Single LUT, Run '+str(i_run))
	print('##########')
	# Se llama a la función .train para el caso de UN CHIP
	results[:,1,i_run], deltaW_info[i_run][1] = train_net.train(filename=task+"_lookup_standard.txt",dataset=task,params=params_standard,n_epochs=Nepochs,\
		activate=activate,alpha=alpha_lut_standard,activate_output=activate_output,learnbias=learnbias,ntset=ntset,ncset=ncset,lr_sched=lr_sched,\
		lr_sched_function=adjust_learning_rate,batchsize=batchsize,loadModelPath=loadModelPaths[i_run][1],saveModelPath=saveModelPaths[i_run][1],sizes=sizes)

	print('##########')
	print('Multi LUT, Run '+str(i_run))
	print('##########')
	# Se llama a la función .train para el caso de VARIOS CHIPS
	results[:,2,i_run], deltaW_info[i_run][2] = train_net.train(filename=task+"_lookup_multi.txt",dataset=task,params=params_multi,n_epochs=Nepochs,\
		activate=activate,alpha=alpha_lut_multi,activate_output=activate_output,learnbias=learnbias,ntset=ntset,ncset=ncset,lr_sched=lr_sched,\
		lr_sched_function=adjust_learning_rate,batchsize=batchsize,loadModelPath=loadModelPaths[i_run][2],saveModelPath=saveModelPaths[i_run][2],sizes=sizes)
	print('Done training '+task+', Multi LUT')

# Calculamos el promedio y la desviación estándar si hicimos más de 1 carrera (Nruns > 1)
results_mean = np.mean(results,2)
results_std = np.std(results,2)

# --- GUARDAR RESULTADOS EN EXCEL (CSV) --- >> sweep results
if saveSweep:
	result_folder = "./sweep_results/"+task+"/"+device_type+"/"
	if result_folder is not None and not os.path.isdir(result_folder):
		os.makedirs(result_folder) # Crea la carpeta si no existe
	# Guarda la tabla de precisión final
	np.savetxt(result_folder+wtmodel+"_noise_mean_all.csv",results_mean,delimiter=",")
	np.savetxt(result_folder+wtmodel+"_noise_std_all.csv",results_std,delimiter=",")

# --- ANÁLISIS ESTADÍSTICO Y GRÁFICAS ---
# Esta parte crea los dibujos (gráficas) para entender el error de los chips
if collect_weight_updates and not periodic_carry:
	diag_folder = "./weight_update_stats/"+task+"/"
	if diag_folder is not None and not os.path.isdir(diag_folder):
		os.makedirs(diag_folder)
	# Archivo de texto para escribir las estadísticas
	fout = diag_folder + "statistics_"+device_type+"_"+wtmodel+"writenoise.txt"
	fileW = open(fout,"w")
	
	# Ciclo para graficar los 3 casos (Numeric, Single, Multi)
	for k in range(3):
		if k == 0:
			lut = 'numeric'
		elif k == 1:
			lut = lookup_table_single
		elif k == 2:
			lut = lookup_table_multi
		
		# Extraemos los datos de lo que queríamos cambiar vs lo que el chip cambió realmente
		diagnostics_k = deltaW_info[0][k]
		target_updates = diagnostics_k['target_updates'] # Lo ideal
		real_updates = diagnostics_k['real_updates']     # Lo real

		N_updates = len(target_updates)
		# Normalizamos los datos para que sean comparables
		target_updates /= (np.std(target_updates)*2)
		real_updates /= (np.std(real_updates)*2)
		update_error = target_updates - real_updates # Calculamos la diferencia (el error)
		
		# Cálculos matemáticos de error promedio y asimetría
		mean_err, std_err = np.mean(update_error), np.std(update_error)
		std_err_pos = np.std(update_error[target_updates>0])
		std_err_neg = np.std(update_error[target_updates<0])
		asym = abs(std_err_pos-std_err_neg)

		# Escribimos los resultados en el archivo de texto
		fileW.write('\nTask: '+str(task))
		fileW.write('\nLook-up table set: '+str(lut))
		fileW.write('\n# updates: '+str(N_updates))
		fileW.write("\nMean error: "+str(mean_err))
		fileW.write("\nSpread error: "+str(std_err))
		fileW.write("\nSpread positive error: "+str(std_err_pos))
		fileW.write("\nSpread negative error: "+str(std_err_neg))
		fileW.write("\nAsymmetry: "+str(asym))
		fileW.write("\n \n")

		# --- CREACIÓN DE LA IMAGEN (PLOTS) ---
		fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6)) # Crea una imagen con dos gráficos
		plt.subplots_adjust(wspace=0.4)
		
		# Gráfico 1: Nube de puntos (Ideal vs Real)
		ax1.scatter(target_updates,real_updates,s=10)
		C = np.maximum(np.max(np.abs(target_updates)),np.max(np.abs(real_updates)))
		x = np.linspace(-C,C,1000)
		ax1.plot(x,x,'--k',linewidth=2) # Línea diagonal (lo perfecto)
		ax1.tick_params(labelsize=14)
		ax1.set_xlabel("Target update",fontsize=14) # Eje X: Lo que pedimos
		ax1.set_ylabel("Real update",fontsize=14)   # Eje Y: Lo que el chip hizo
		ax1.set_xlim(-C,C); ax1.set_ylim(-C,C)
		
		# Gráfico 2: Histograma (Distribución del error)
		ax2.hist(update_error,bins=100)
		ax2.set_xlabel('Update error',fontsize=14) 
		ax2.set_ylabel('Probability density',fontsize=14)
		ax2.tick_params(labelsize=14)
		D = np.max(np.abs(update_error))
		if D > 0:
			ax2.set_xlim(-D,D)
		else:
			ax2.set_xlim(-1,1)
		
		# Guardamos la imagen en la carpeta
		save_filename = diag_folder + "innercore_update_error_"+wtmodel+"_"+lut+".png"
		fig.savefig(save_filename,dpi=600,bbox_inches='tight')

	fileW.close() # Cerramos el archivo de texto al finalizar
    