#
# run_experiments.py
# IE0499 Proyecto Electrico I-2026 — Universidad de Costa Rica
#
# CONTROL: para omitir un experimento , poner su flag en False.
# =============================================================================

import numpy as np
import sys, os, time
import matplotlib
matplotlib.use('Agg')   # sin ventana GUI, necesario en headless / scripts
import matplotlib.pyplot as plt
from training_net import train_neural_net
from lr_schedule import adjust_learning_rate

np.set_printoptions(precision=4, suppress=True)

# =============================================================================
# CONTROL DE EXPERIMENTOS
# =============================================================================
RUN_EXP0 = True   # Baseline (replica exacta de la primera run)
RUN_EXP1 = True   # batchsize = 50  (mini-batch intermedio)
RUN_EXP2 = True   # batchsize = 200 (mini-batch grande, actualizaciones suaves)
RUN_EXP3 = True   # Alpha x4  (tasa agresiva: 0.04 / 0.01 / 0.004)
RUN_EXP4 = True   # Alpha x0.25 (tasa conservadora: 0.0025 / 0.000625 / 0.00025)
RUN_EXP5 = True   # device_type = "TaOx" (cambio de material memristivo)
RUN_EXP6 = True   # wtmodel = "OFFSET" (codificacion offset vs balanced)
RUN_EXP7 = False  # stochastic_updates = True
                  # ADVERTENCIA: puede aumentar mucho el tiempo; activar con precaucion

# =============================================================================
# PARAMETROS FIJOS (no varian entre experimentos)
# =============================================================================
FIXED = dict(
    useGPU              = True,
    gpu_num             = 0,
    task                = "mnist",
    sizes               = (784, 300, 10),
    Nepochs             = 1,       # 1 epoca por limitacion de hardware
    Nruns               = 1,
    ntset               = 0,       # 0 = sin truncacion (60 000 ejemplos)
    ncset               = 0,       # 0 = sin truncacion (10 000 ejemplos)
    activate            = "SIGMOID",
    activate_output     = "SOFTMAX",
    a2dmodel            = "NONE",
    learnbias           = (True, True),
    periodic_carry      = False,
    pc_Nslices          = 2,
    pc_number_base      = 16,
    saveCrossSimModels  = True,
    collect_weight_updates = True,
    Nupdates_total      = 100000,
)

# Alphas por defecto (mnist, de MLP_training.py original)
DEFAULT_ALPHA = dict(
    alpha_numeric      = 0.01,
    alpha_lut_standard = 0.0025,
    alpha_lut_multi    = 0.001,
)

# Baseline de los parametros que SI varian entre experimentos
BASELINE = dict(
    device_type        = "DWMTJ",
    batchsize          = 10,        # valor original
    wtmodel            = "BALANCED",
    lr_sched           = True,
    stochastic_updates = False,
    **DEFAULT_ALPHA,
)

# =============================================================================
# DEFINICION DE EXPERIMENTOS
# Formato: (nombre, descripcion, dict de overrides sobre BASELINE)
#
#   batchsize=10 -> baseline (6 000 actualizaciones)
#   batchsize=50 -> 1 200 actualizaciones, deberia ser mas rapido que baseline
#   batchsize=200-> 300 actualizaciones, el mas rapido de todos
# =============================================================================
EXPERIMENTS = []

if RUN_EXP0:
    EXPERIMENTS.append((
        "exp0_baseline",
        "Baseline: batchsize=10, DWMTJ STT 300K, BALANCED, alpha nominal",
        {}
    ))

if RUN_EXP1:
    EXPERIMENTS.append((
        "exp1_batchsize50",
        "batchsize=50: mini-batch intermedio (5x baseline), menos consultas a LUT por epoca",
        {"batchsize": 50}
    ))

if RUN_EXP2:
    EXPERIMENTS.append((
        "exp2_batchsize200",
        "batchsize=200: mini-batch grande (20x baseline), actualizaciones muy suaves",
        {"batchsize": 200}
    ))

if RUN_EXP3:
    EXPERIMENTS.append((
        "exp3_alpha_high",
        "Alpha x4 agresivo: numeric=0.04, single=0.01, multi=0.004",
        {"alpha_numeric": 0.04,
         "alpha_lut_standard": 0.01,
         "alpha_lut_multi": 0.004}
    ))

if RUN_EXP4:
    EXPERIMENTS.append((
        "exp4_alpha_low",
        "Alpha x0.25 conservador: numeric=0.0025, single=0.000625, multi=0.00025",
        {"alpha_numeric": 0.0025,
         "alpha_lut_standard": 0.000625,
         "alpha_lut_multi": 0.00025}
    ))

if RUN_EXP5:
    EXPERIMENTS.append((
        "exp5_TaOx",
        "device_type=TaOx: ReRAM Sandia 2016, mismo batchsize y alphas que baseline",
        {"device_type": "TaOx"}
    ))

if RUN_EXP6:
    EXPERIMENTS.append((
        "exp6_wtmodel_OFFSET",
        "wtmodel=OFFSET: un core por peso con conductancia minima de referencia (vs BALANCED que usa dos cores)",
        {"wtmodel": "OFFSET"}
    ))

if RUN_EXP7:
    EXPERIMENTS.append((
        "exp7_stochastic",
        "stochastic_updates=True: redondeo estocastico en la actualizacion de pesos",
        {"stochastic_updates": True}
    ))

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def resolve_lut_paths(device_type):
    """Devuelve (lookup_single, lookup_multi) segun el material."""
    if device_type == "TaOx":
        return "TaOx", "TaOx_medium_set"
    elif device_type == "ENODe":
        return "ENODe", "ENODe_multi"
    elif device_type == "DWMTJ":
        return "DWMTJ_STT_300K", "DWMTJ_STT_300K_multi"
    else:
        raise ValueError(f"device_type '{device_type}' no reconocido")


def ensure_dirs(base):
    """Crea la estructura de carpetas para un experimento."""
    for sub in ["console_output", "cross_sim_models",
                "sweep_results", "weight_update_stats"]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)


def save_diagnostics(deltaW_info, task, device_type, wtmodel,
                     lut_single, lut_multi, outbase):
    """Genera histogramas y metricas de error de actualizacion de hardware."""
    diag_folder = os.path.join(outbase, "weight_update_stats", task, "")
    os.makedirs(diag_folder, exist_ok=True)
    fout = diag_folder + f"statistics_{device_type}_{wtmodel}writenoise.txt"

    lut_names = ['numeric', lut_single, lut_multi]

    with open(fout, "w") as fileW:
        for k in range(3):
            diag_k = deltaW_info[0][k]
            if diag_k is None:
                continue

            if diag_k.get('target_updates') is None or diag_k.get('real_updates') is None:
                continue

            t_upd = diag_k['target_updates'].copy()
            r_upd = diag_k['real_updates'].copy()

            N = len(t_upd)
            std_t = np.std(t_upd)
            std_r = np.std(r_upd)
            if std_t > 0: t_upd /= (std_t * 2)
            if std_r > 0: r_upd /= (std_r * 2)

            err   = t_upd - r_upd
            m_err = np.mean(err)
            s_err = np.std(err)
            pos   = t_upd > 0
            neg   = t_upd < 0
            s_pos = np.std(err[pos]) if pos.any() else 0.0
            s_neg = np.std(err[neg]) if neg.any() else 0.0
            asym  = abs(s_pos - s_neg)

            fileW.write(f'\nTask: {task}')
            fileW.write(f'\nLook-up table set: {lut_names[k]}')
            fileW.write(f'\n# updates: {N}')
            fileW.write(f'\nMean error: {m_err}')
            fileW.write(f'\nSpread error: {s_err}')
            fileW.write(f'\nSpread positive error: {s_pos}')
            fileW.write(f'\nSpread negative error: {s_neg}')
            fileW.write(f'\nAsymmetry: {asym}')
            fileW.write('\n \n')

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            plt.subplots_adjust(wspace=0.4)

            ax1.scatter(t_upd, r_upd, s=10)
            C = max(np.max(np.abs(t_upd)), np.max(np.abs(r_upd)))
            x = np.linspace(-C, C, 1000)
            ax1.plot(x, x, '--k', linewidth=2)
            ax1.set_xlabel("Target update", fontsize=14)
            ax1.set_ylabel("Real update",   fontsize=14)
            ax1.set_xlim(-C, C); ax1.set_ylim(-C, C)
            ax1.tick_params(labelsize=14)

            ax2.hist(err, bins=100, density=True)
            ax2.set_xlabel('Update error',        fontsize=14)
            ax2.set_ylabel('Probability density', fontsize=14)
            ax2.tick_params(labelsize=14)
            D = np.max(np.abs(err))
            ax2.set_xlim(-D if D > 0 else -1, D if D > 0 else 1)

            fig_path = diag_folder + f"innercore_update_error_{wtmodel}_{lut_names[k]}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)


def run_single_experiment(exp_name, description, param_overrides):
    """
    Ejecuta un experimento completo.
    Retorna dict con resumen de resultados para la tabla final.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  EXPERIMENTO: {exp_name}")
    print(f"  {description}")
    print(f"{sep}\n")

    # Parametros efectivos = baseline + overrides
    p = {**BASELINE, **param_overrides}

    task             = FIXED['task']
    sizes            = FIXED['sizes']
    Nepochs          = FIXED['Nepochs']
    Nruns            = FIXED['Nruns']
    ntset            = FIXED['ntset']
    ncset            = FIXED['ncset']
    activate         = FIXED['activate']
    activate_output  = FIXED['activate_output']
    a2dmodel         = FIXED['a2dmodel']
    learnbias        = FIXED['learnbias']
    periodic_carry   = FIXED['periodic_carry']
    pc_Nslices       = FIXED['pc_Nslices']
    pc_number_base   = FIXED['pc_number_base']
    Nupdates_total   = FIXED['Nupdates_total']
    collect_wu       = FIXED['collect_weight_updates']
    saveModels       = FIXED['saveCrossSimModels']
    useGPU           = FIXED['useGPU']

    device_type  = p['device_type']
    batchsize    = p['batchsize']
    wtmodel      = p['wtmodel']
    lr_sched     = p['lr_sched']
    stoch        = p['stochastic_updates']
    alpha_num    = p['alpha_numeric']
    alpha_std    = p['alpha_lut_standard']
    alpha_multi  = p['alpha_lut_multi']

    # ENODe no soporta GPU 

    # ENODe no soporta GPU 
    if device_type == "ENODe":
        useGPU = False

    # OFFSET no es compatible con CuPy (usa numpy.insert)
    if wtmodel == "OFFSET":
        useGPU = False

    if useGPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FIXED['gpu_num'])

    lut_single, lut_multi = resolve_lut_paths(device_type)

    # Carpetas de salida
    outbase = os.path.join("results", exp_name)
    ensure_dirs(outbase)
    outdir_console = os.path.join(outbase, "console_output", "")

    # Contenedores
    results     = np.zeros((Nepochs, 3, Nruns))
    deltaW_info = [[None, None, None] for _ in range(Nruns)]
    diagParams  = [collect_wu, Nupdates_total]

    types = ["numeric", "lookup_standard", "lookup_multi"]
    saveModelPaths = [[None] * 3 for _ in range(Nruns)]
    if saveModels:
        models_dir = os.path.join(outbase, "cross_sim_models")
        for j in range(3):
            for i in range(Nruns):
                saveModelPaths[i][j] = os.path.join(
                    models_dir, f"MLP_{task}_{types[j]}_run{i}.npz")

    t0 = time.time()

    for i_run in range(Nruns):

        train_net = train_neural_net(outdir_console)

        params_numeric = train_net.set_params(
            task=task, lookup_table=None, a2dmodel=a2dmodel,
            stochastic_updates=stoch, wtmodel=wtmodel, learnbias=learnbias,
            diagnosticParams=diagParams, useGPU=useGPU)

        params_standard = train_net.set_params(
            task=task, lookup_table=lut_single, a2dmodel=a2dmodel,
            stochastic_updates=stoch, wtmodel=wtmodel, learnbias=learnbias,
            diagnosticParams=diagParams, useGPU=useGPU,
            periodic_carry=periodic_carry,
            pc_number_base=pc_number_base, pc_Nslices=pc_Nslices)

        params_multi = train_net.set_params(
            task=task, lookup_table=lut_multi, a2dmodel=a2dmodel,
            stochastic_updates=stoch, wtmodel=wtmodel, learnbias=learnbias,
            diagnosticParams=diagParams, useGPU=useGPU,
            periodic_carry=periodic_carry,
            pc_number_base=pc_number_base, pc_Nslices=pc_Nslices)

        # --- Numeric ---
        print(f"  [Run {i_run}] Numeric (ideal)...")
        results[:, 0, i_run], deltaW_info[i_run][0] = train_net.train(
            filename=task + "_numeric.txt", dataset=task,
            params=params_numeric, n_epochs=Nepochs,
            activate=activate, alpha=alpha_num,
            activate_output=activate_output, learnbias=learnbias,
            ntset=ntset, ncset=ncset,
            lr_sched=lr_sched, lr_sched_function=adjust_learning_rate,
            batchsize=batchsize,
            loadModelPath=None, saveModelPath=saveModelPaths[i_run][0],
            sizes=sizes)

        # --- Single LUT ---
        print(f"  [Run {i_run}] Single LUT ({lut_single})...")
        results[:, 1, i_run], deltaW_info[i_run][1] = train_net.train(
            filename=task + "_lookup_standard.txt", dataset=task,
            params=params_standard, n_epochs=Nepochs,
            activate=activate, alpha=alpha_std,
            activate_output=activate_output, learnbias=learnbias,
            ntset=ntset, ncset=ncset,
            lr_sched=lr_sched, lr_sched_function=adjust_learning_rate,
            batchsize=batchsize,
            loadModelPath=None, saveModelPath=saveModelPaths[i_run][1],
            sizes=sizes)

        # --- Multi LUT ---
        print(f"  [Run {i_run}] Multi LUT ({lut_multi})...")
        results[:, 2, i_run], deltaW_info[i_run][2] = train_net.train(
            filename=task + "_lookup_multi.txt", dataset=task,
            params=params_multi, n_epochs=Nepochs,
            activate=activate, alpha=alpha_multi,
            activate_output=activate_output, learnbias=learnbias,
            ntset=ntset, ncset=ncset,
            lr_sched=lr_sched, lr_sched_function=adjust_learning_rate,
            batchsize=batchsize,
            loadModelPath=None, saveModelPath=saveModelPaths[i_run][2],
            sizes=sizes)

    elapsed = time.time() - t0

    results_mean = np.mean(results, axis=2)
    results_std  = np.std(results,  axis=2)

    res_folder = os.path.join(outbase, "sweep_results", task, device_type, "")
    os.makedirs(res_folder, exist_ok=True)
    np.savetxt(res_folder + wtmodel + "_noise_mean_all.csv", results_mean, delimiter=",")
    np.savetxt(res_folder + wtmodel + "_noise_std_all.csv",  results_std,  delimiter=",")

    if collect_wu and not periodic_carry and wtmodel != "OFFSET":
        save_diagnostics(deltaW_info, task, device_type, wtmodel,
                         lut_single, lut_multi, outbase)

    summary = {
        "exp_name"    : exp_name,
        "description" : description,
        "params"      : dict(p),
        "acc_numeric" : float(results_mean[-1, 0]),
        "acc_single"  : float(results_mean[-1, 1]),
        "acc_multi"   : float(results_mean[-1, 2]),
        "time_s"      : round(elapsed, 1),
    }

    print(f"\n  OK {exp_name}  |  "
          f"Numeric={summary['acc_numeric']:.4f}  "
          f"Single={summary['acc_single']:.4f}  "
          f"Multi={summary['acc_multi']:.4f}  "
          f"({summary['time_s']} s)\n")

    return summary


def print_and_save_table(summaries):
    """Imprime la tabla comparativa y la guarda en CSV."""
    print("\n" + "=" * 82)
    print("  TABLA COMPARATIVA DE TODOS LOS EXPERIMENTOS")
    print("=" * 82)
    print(f"  {'Experimento':<28} {'Numeric':>9} {'SingleLUT':>10} "
          f"{'MultiLUT':>9} {'Tiempo':>9}")
    print("  " + "-" * 68)
    for s in summaries:
        print(f"  {s['exp_name']:<28} "
              f"{s['acc_numeric']:>9.4f} "
              f"{s['acc_single']:>10.4f} "
              f"{s['acc_multi']:>9.4f} "
              f"{s['time_s']:>8.0f}s")
    print("=" * 82)

    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "comparacion_experimentos.csv")
    with open(csv_path, "w") as f:
        f.write("experimento,descripcion,"
                "acc_numeric,acc_single_lut,acc_multi_lut,tiempo_s,"
                "device_type,batchsize,wtmodel,lr_sched,"
                "stochastic,alpha_num,alpha_std,alpha_multi\n")
        for s in summaries:
            p = s.get('params', {})

            f.write(
                f"{s['exp_name']},"
                f"\"{s['description']}\","
                f"{s['acc_numeric']:.6f},"
                f"{s['acc_single']:.6f},"
                f"{s['acc_multi']:.6f},"
                f"{s['time_s']},"
                f"{p.get('device_type', 'N/A')},"
                f"{p.get('batchsize', 'N/A')},"
                f"{p.get('wtmodel', 'N/A')},"
                f"{p.get('lr_sched', 'N/A')},"
                f"{p.get('stochastic_updates', 'N/A')},"
                f"{p.get('alpha_numeric', 'N/A')},"
                f"{p.get('alpha_lut_standard', 'N/A')},"
                f"{p.get('alpha_lut_multi', 'N/A')}\n"
            )
    print(f"\n  Tabla guardada en: {csv_path}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    print("\n" + "#" * 70)
    print("  CrossSim 2.0 - Experimentos sistematicos")
    print("  IE0499 Proyecto Electrico I-2026, UCR - RTX 4060")
    print("#" * 70)
    print(f"  Experimentos a ejecutar: {len(EXPERIMENTS)}\n")
    for i, (name, desc, _) in enumerate(EXPERIMENTS):
        print(f"  [{i}] {name}")
        print(f"      {desc}")
    print()

    t_global = time.time()
    all_summaries = []

    for exp_name, description, overrides in EXPERIMENTS:
        try:
            s = run_single_experiment(exp_name, description, overrides)
            all_summaries.append(s)
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] '{exp_name}' fallo: {e}")
            traceback.print_exc()
            all_summaries.append({
                "exp_name"    : exp_name,
                "description" : description,
                "params"      : {},
                "acc_numeric" : float('nan'),
                "acc_single"  : float('nan'),
                "acc_multi"   : float('nan'),
                "time_s"      : 0.0,
            })

    print_and_save_table(all_summaries)
    total = (time.time() - t_global) / 60
    print(f"\n  Tiempo total de ejecucion: {total:.1f} min")
    print("  Resultados en: cross-sim/training/results/\n")