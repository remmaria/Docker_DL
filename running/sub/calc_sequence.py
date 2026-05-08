import numpy as np
import sys


def calc_sequence(bval_file):

    bvals = np.loadtxt(bval_file)

    targets = [0,500,700,750,800,1000,1500,2000]
    tolerance=15

    val_tolerance = []
    for v in bvals:
        ok = False
        for target in targets:
            if target - tolerance <= v <= target + tolerance:
                val_tolerance.append(target)
                ok = True
                break
        if ok == False:
            print('ERRO', v,flush=True, file=sys.stderr)
    if len(bvals) != len(val_tolerance):
        print("ERROOO!!!!", len(bvals),len(val_tolerance),flush=True, file=sys.stderr)

    # Obter valores únicos e contá-los
    unique_values = sorted(set(val_tolerance))  # Ordena para facilitar a leitura
    #print(f"Valores únicos ajustados: {unique_values}", flush=True, file=sys.stderr)

    dic_seq = {}
    for bval in unique_values:
        dic_seq[bval] = val_tolerance.count(bval) 


    dic_seq_str=""
    for k in dic_seq.keys():
        dic_seq_str+=f"{k}-{dic_seq[k]}_"
    dic_seq_str=dic_seq_str[:-1]

    return bvals, dic_seq, dic_seq_str, unique_values, val_tolerance