"""
3D Autoencoder Recorrente para super-resolucao angular em dMRI -- reproducao
FIEL da arquitetura oficial de Lyon et al. 2022 (arXiv:2203.15598,
github.com/m-lyon/dMRI-RCNN), portada manualmente de TensorFlow/Keras
(dmri_rcnn/core/model/{autoencoder,layers}.py, lido direto do repo oficial)
para PyTorch.

Isto substitui a versao anterior (ConvGRU customizada + 1-2 blocos simples
por estagio, ~16-32 canais, injecao do embedding de direcao so no comeco
do decoder) por uma replica estrutural do modelo `get_3d_encoder` /
`get_3d_decoder` do paper:
  - blocos multi-ramo (kernels 1, 2 e 3 EM PARALELO, concatenados entre si
    e com a entrada do bloco -- padrao tipo Inception/DenseNet) com as
    MESMAS contagens de canal do paper (104-668 no encoder e no decoder).
  - InstanceNorm3d no primeiro estagio (`init`/`ecv11..13`), BatchNorm3d
    nos seguintes -- dos dois lados, encoder e decoder.
  - ativacao swish (nn.SiLU) em tudo, exceto a saida final (`dc4`, ReLU
    sem norm) e o bloco de compressao `dcv3` (sem norm, ainda swish).
  - reinjecao do bvec UNITARIO (sem transformacao, so concatenado, 3
    canais) a cada estagio (RepeatBVector), tanto no encoder quanto no
    decoder -- e nao mais um "embedding" custom de 6 dims com bval
    reinjetado so uma vez.
  - agregacao da sequencia de direcoes de ENTRADA por uma ConvLSTM3D
    (kernel 1x1x1, `return_sequences=False`), nao mais uma ConvGRU
    customizada.
  - bval NAO entra na condicao do modelo (o paper condiciona so no bvec
    unitario -- dentro de uma shell so, o bval e constante, entao nao
    carrega informacao nenhuma pro modelo). `input_bvals`/`target_bvals`/
    `b_ref` continuam na assinatura de `RCAE.forward` por compatibilidade
    com scripts/04_train_rcae.py, mas sao ignorados aqui dentro.

Diferencas de fidelidade que AINDA PERMANECEM (documentadas, nao
escondidas -- sinalize se quiser fechar tambem numa proxima rodada):
  - "SAME" padding do TF pra kernel PAR (k=2) usa padding assimetrico (0
    antes, 1 depois do eixo) -- replicado explicitamente em
    `_same_pad_3d`, entao isso NAO e uma diferenca, so documentando que
    exigiu atencao (padding simetrico ingenuo, `padding=k//2`, teria dado
    output maior que o input pra k par).
  - a ativacao recorrente da ConvLSTM3D do Keras e 'hard_sigmoid' por
    default; usamos sigmoid comum (torch nao tem hard_sigmoid nativo
    identico) -- diferenca numerica pequena, nao deve mudar o
    comportamento qualitativo.
  - inicializacao de pesos: usamos `kaiming_uniform_` (proximo de
    `he_uniform` do Keras, mesma familia, nao bit-a-bit identico).
  - nao reproduzimos os modelos alternativos do paper (`get_1d_*`, versao
    "pointwise" mais leve) nem carregamento de pesos pre-treinados
    (`weights=`) -- treinamos do zero nos seus dados, como ja vinha sendo
    feito.

ATENCAO: os canais deste modelo sao bem maiores que a versao anterior
(centenas de canais por bloco) -- espere uso de VRAM/tempo por epoca mais
alto que antes. Se estourar memoria de GPU, reduza --batch-size antes de
mexer na arquitetura (mudar os canais quebraria a fidelidade que voce
pediu).

Requer PyTorch (nao disponivel neste ambiente de desenvolvimento -- revisado
manualmente, testado apenas por compilacao de sintaxe; validar no cluster
com `python -m model.rcae`, smoke test no fim do arquivo).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_pad_3d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Replica o padding 'SAME' do TensorFlow (stride=1, dilation=1): o
    padding total (kernel_size - 1) e dividido de forma ASSIMETRICA quando
    impar -- menos ANTES, mais DEPOIS de cada eixo espacial. Pra
    kernel_size=2 isso da pad_before=0, pad_after=1 (um Conv3d com
    `padding=kernel_size//2` simetrico, que seria o "jeito ingenuo", daria
    output MAIOR que o input pra kernel par -- por isso o padding manual
    aqui em vez de usar o argumento `padding=` do nn.Conv3d)."""
    total = kernel_size - 1
    pad_before = total // 2
    pad_after = total - pad_before
    return F.pad(x, [pad_before, pad_after] * 3)  # (W,W,H,H,D,D) -- mesmo valor nos 3 eixos


class SamePadConv3D(nn.Module):
    """Conv3d com padding 'SAME' manual (ver _same_pad_3d) + init
    kaiming_uniform (~he_uniform do Keras) -- replica
    DistributedConv3D._get_conv_layer do paper (`padding='same'`,
    `kernel_initializer=he_uniform()`)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, padding=0)
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(_same_pad_3d(x, self.kernel_size))


class DistributedConv3D(nn.Module):
    """Equivalente a `layers.TimeDistributed(Conv3D)` do paper: aplica a
    MESMA conv3d a cada "passo de tempo" (aqui, cada direcao de
    entrada/alvo) independentemente -- dobra (B,T,C,D,H,W) em
    (B*T,C,D,H,W), aplica, desdobra de volta. Ordem conv -> ativacao ->
    norm (igual ao `call()` de DistributedConv3D em
    dmri_rcnn/core/model/layers.py -- norm depois da ativacao, nao antes,
    detalhe facil de inverter por engano)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 norm: str | None = None, activation: str | None = "swish"):
        super().__init__()
        assert norm in (None, "instance", "batch"), f"norm invalido: {norm}"
        assert activation in (None, "swish", "relu"), f"activation invalida: {activation}"
        self.conv = SamePadConv3D(in_ch, out_ch, kernel_size)
        if activation == "swish":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = None
        if norm == "instance":
            self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        elif norm == "batch":
            self.norm = nn.BatchNorm3d(out_ch)
        else:
            self.norm = None

    def forward(self, x):
        # x: (B, T, C_in, D, H, W)
        b, t = x.shape[0], x.shape[1]
        x = x.reshape(b * t, *x.shape[2:])
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        if self.norm is not None:
            x = self.norm(x)
        x = x.reshape(b, t, *x.shape[1:])
        return x


def _repeat_bvec(bvec: torch.Tensor, spatial_shape) -> torch.Tensor:
    """bvec: (B, T, 3) -> (B, T, 3, D, H, W), broadcast espacial --
    equivalente a `RepeatBVector` do paper (concatena o bvec cru, sem
    nenhuma transformacao, em cada estagio conv)."""
    b, t, c = bvec.shape
    x = bvec.view(b, t, c, 1, 1, 1)
    return x.expand(b, t, c, *spatial_shape)


def _repeat_state(state: torch.Tensor, n: int) -> torch.Tensor:
    """state: (B, C, D, H, W) -> (B, n, C, D, H, W) -- equivalente a
    `RepeatTensor` do paper, repete o hidden state final do encoder pra
    cada uma das n direcoes-alvo antes do decoder processar em paralelo."""
    return state.unsqueeze(1).expand(-1, n, -1, -1, -1, -1)


class ConvLSTMCell3D(nn.Module):
    """Celula ConvLSTM 3D (kernel 1x1x1, igual ao paper:
    `layers.ConvLSTM3D(lstm_size, 1)`). Ativacao tanh pra atualizacao de
    celula/output, sigmoid pros gates (Keras usa 'hard_sigmoid' por
    default pros gates -- aproximado aqui com sigmoid comum, ver nota de
    fidelidade no topo do arquivo)."""

    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 1):
        super().__init__()
        pad = kernel_size // 2  # kernel=1 -> pad=0 (impar, sem assimetria a considerar)
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv3d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad)

    def forward(self, x_t, state):
        h_prev, c_prev = state
        gates = self.conv(torch.cat([x_t, h_prev], dim=1))
        i, f, g, o = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM3D(nn.Module):
    """Roda ConvLSTMCell3D ao longo do eixo T (sequencia de direcoes de
    entrada) e devolve so o hidden state FINAL -- equivalente a
    `return_sequences=False` (default do Keras `ConvLSTM3D`, e o que o
    paper usa: o encoder devolve um unico "estado" de contexto, nao uma
    sequencia)."""

    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 1):
        super().__init__()
        self.cell = ConvLSTMCell3D(in_ch, hidden_ch, kernel_size)
        self.hidden_ch = hidden_ch

    def forward(self, x):
        # x: (B, T, C, D, H, W)
        b, t, _, d, h, w = x.shape
        state = (x.new_zeros(b, self.hidden_ch, d, h, w),
                  x.new_zeros(b, self.hidden_ch, d, h, w))
        for step in range(t):
            state = self.cell(x[:, step], state)
        return state[0]  # (B, hidden_ch, D, H, W)


class Encoder3D(nn.Module):
    """Replica `get_3d_encoder` (dmri_rcnn/core/model/autoencoder.py).
    Contagens de canal EXATAMENTE as do paper (nao sao hiperparametros
    livres -- fixas de proposito, pra manter a fidelidade pedida)."""

    def __init__(self, lstm_size: int = 48):
        super().__init__()
        # qspace_tensor = concat(imagem, bvec) = 1 + 3 = 4 canais
        self.init = DistributedConv3D(4, 200, 1, norm="instance")
        self.ecv11 = DistributedConv3D(200, 104, 1, norm="instance")
        self.ecv12 = DistributedConv3D(200, 200, 2, norm="instance")
        self.ecv13 = DistributedConv3D(200, 72, 3, norm="instance")
        # conv1 = concat(ecv11, ecv12, ecv13, qspace) -> 104+200+72+4 = 380
        self.ecv21 = DistributedConv3D(380, 280, 1, norm="batch")
        self.ecv22 = DistributedConv3D(380, 240, 2, norm="batch")
        self.ecv23 = DistributedConv3D(380, 144, 3, norm="batch")
        # conv2 = concat(ecv21, ecv22, ecv23, qspace) -> 280+240+144+4 = 668
        self.ecvl1 = DistributedConv3D(668, 32, 1, norm="batch")
        self.ecvl2 = DistributedConv3D(32, 88, 1, norm="batch")
        self.lstm = ConvLSTM3D(88, lstm_size, kernel_size=1)

    def forward(self, input_vols: torch.Tensor, input_bvecs: torch.Tensor) -> torch.Tensor:
        # input_vols: (B, N_in, 1, D, H, W); input_bvecs: (B, N_in, 3)
        spatial = input_vols.shape[-3:]
        bvec_map = _repeat_bvec(input_bvecs, spatial)          # (B,N_in,3,D,H,W)
        qspace = torch.cat([input_vols, bvec_map], dim=2)      # (B,N_in,4,D,H,W)

        init = self.init(qspace)
        e11, e12, e13 = self.ecv11(init), self.ecv12(init), self.ecv13(init)
        conv1 = torch.cat([e11, e12, e13, qspace], dim=2)

        e21, e22, e23 = self.ecv21(conv1), self.ecv22(conv1), self.ecv23(conv1)
        conv2 = torch.cat([e21, e22, e23, qspace], dim=2)

        latent = self.ecvl2(self.ecvl1(conv2))
        return self.lstm(latent)  # (B, lstm_size, D, H, W)


class Decoder3D(nn.Module):
    """Replica `get_3d_decoder` (dmri_rcnn/core/model/autoencoder.py)."""

    def __init__(self, lstm_size: int = 48):
        super().__init__()
        self.dcvl1 = DistributedConv3D(lstm_size + 3, 176, 1, norm="batch")
        self.dcvl2 = DistributedConv3D(176, 224, 1, norm="batch")
        # latente = concat(dcvl2, bvec) -> 224+3 = 227
        self.dcv11 = DistributedConv3D(227, 240, 1, norm="batch")
        self.dcv12 = DistributedConv3D(227, 256, 2, norm="batch")
        self.dcv13 = DistributedConv3D(227, 136, 3, norm="batch")
        # conv1 = concat(dcv11, dcv12, dcv13, bvec) -> 240+256+136+3 = 635
        self.dcv21 = DistributedConv3D(635, 176, 1, norm="batch")
        self.dcv22 = DistributedConv3D(635, 136, 2, norm="batch")
        self.dcv23 = DistributedConv3D(635, 88, 3, norm="batch")
        # conv2 = concat(dcv21, dcv22, dcv23, bvec) -> 176+136+88+3 = 403
        self.dcv3 = DistributedConv3D(403, 16, 1, norm=None)
        self.dc4 = DistributedConv3D(16, 1, 1, norm=None, activation="relu")

    def forward(self, state: torch.Tensor, target_bvecs: torch.Tensor) -> torch.Tensor:
        # state: (B, lstm_size, D, H, W); target_bvecs: (B, N_out, 3)
        n_out = target_bvecs.shape[1]
        spatial = state.shape[-3:]
        state_seq = _repeat_state(state, n_out)                # (B,N_out,lstm_size,D,H,W)
        bvec_map = _repeat_bvec(target_bvecs, spatial)          # (B,N_out,3,D,H,W)
        latent = torch.cat([state_seq, bvec_map], dim=2)

        latent = self.dcvl2(self.dcvl1(latent))
        latent = torch.cat([latent, bvec_map], dim=2)

        d11, d12, d13 = self.dcv11(latent), self.dcv12(latent), self.dcv13(latent)
        conv1 = torch.cat([d11, d12, d13, bvec_map], dim=2)

        d21, d22, d23 = self.dcv21(conv1), self.dcv22(conv1), self.dcv23(conv1)
        conv2 = torch.cat([d21, d22, d23, bvec_map], dim=2)

        return self.dc4(self.dcv3(conv2))  # (B, N_out, 1, D, H, W)


class RCAE(nn.Module):
    """Autoencoder recorrente completo -- Encoder3D + Decoder3D acima,
    replica estrutural do `get_3d_autoencoder` do paper.

    Uso:
        model = RCAE(lstm_size=48)
        out = model(input_vols, input_bvecs, input_bvals, target_bvecs, target_bvals, b_ref)
    input_vols: (B, N_in, 1, D, H, W); input_bvecs: (B, N_in, 3)
    target_bvecs: (B, N_out, 3)
    retorna: (B, N_out, 1, D, H, W)

    `input_bvals`/`target_bvals`/`b_ref` sao aceitos e IGNORADOS (ver nota
    no topo do arquivo) -- mantidos so pra nao quebrar a assinatura usada
    em scripts/04_train_rcae.py.
    """

    def __init__(self, lstm_size: int = 48):
        super().__init__()
        self.encoder = Encoder3D(lstm_size=lstm_size)
        self.decoder = Decoder3D(lstm_size=lstm_size)

    def forward(self, input_vols, input_bvecs, input_bvals, target_bvecs, target_bvals, b_ref=None):
        state = self.encoder(input_vols, input_bvecs)
        return self.decoder(state, target_bvecs)


def _smoke_test():
    """Forward pass com tensores pequenos aleatorios, so pra checar shapes
    (os canais internos sao os do paper, fixos -- so patch/lstm_size/N_in/
    N_out variam aqui pra o teste rodar rapido). Rodar no cluster (onde
    torch esta instalado): python -m model.rcae
    """
    torch.manual_seed(0)
    b, n_in, n_out, d, h, w = 1, 6, 4, 10, 10, 10
    model = RCAE(lstm_size=8)  # lstm_size pequeno so pra o smoke test rodar rapido/leve
    input_vols = torch.rand(b, n_in, 1, d, h, w)
    input_bvecs = torch.randn(b, n_in, 3)
    input_bvecs = input_bvecs / input_bvecs.norm(dim=-1, keepdim=True)
    input_bvals = torch.full((b, n_in), 1000.0)
    target_bvecs = torch.randn(b, n_out, 3)
    target_bvecs = target_bvecs / target_bvecs.norm(dim=-1, keepdim=True)
    target_bvals = torch.full((b, n_out), 1000.0)

    out = model(input_vols, input_bvecs, input_bvals, target_bvecs, target_bvals, b_ref=1000.0)
    expected = (b, n_out, 1, d, h, w)
    assert out.shape == expected, f"shape mismatch: {out.shape} != {expected}"
    print("smoke test OK, output shape:", tuple(out.shape))


if __name__ == "__main__":
    _smoke_test()