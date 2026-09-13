import pandas as pd

df = pd.read_csv("/ix1/tibrahim/rmm270/Docker_DL/Lyon/work_dir/rcae_checkpoints/shell1000_n16_sh/runs/3828686_0/batch_log.csv")
resumo = (df.groupby(["epoch", "split"])[["loss_signal", "loss_angular"]]
            .mean()
            .reset_index())
print(resumo)