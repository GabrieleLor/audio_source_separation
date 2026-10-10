#  Audio Source Separation

Progetto finale del corso di studi si  Trattamento dei dati multimediali.

Isolamento della traccia vocale da un brano musicale tramite una rete U-Net che lavora su spettrogrammi, addestrata sul dataset MUSDB18.

## Indice

- [Idea del progetto](#idea-del-progetto)
- [Struttura del repo](#struttura-del-repository)
- [Setup — Google Colab (consigliato per GPU)](#setup--google-colab-consigliato)
- [Setup — Locale con Anaconda ](#setup--locale-con-anaconda)
- [Esecuzione](#esecuzione)
- [Dettagli tecnici del progetto](#dettagli-tecnici)
- [Risultati](#risultati)
- [Limiti ed eventuali possibili miglioramenti](#limiti-e-possibili-miglioramenti)

## Idea del progetto

L'approccio è a maschera su spettrogramma, le fase del progetto sono:

1. L'audio (mix) viene trasformato in spettrogramma con la STFT (Short-Time Fourier Transform).
2. Una U-Net impara a predire, per ogni punto tempo-frequenza, quanto appartiene alla voce (valore tra 0 e 1).
3. La maschera moltiplica lo spettrogramma di magnitudine del mix, ottenendo la magnitudine della voce stimata.
4. Si torna all'audio con la ISTFT, riusando la fase del mix originale (il modello non la predice).

Il training è supervisionato: MUSDB18 fornisce, per ogni brano, il mix e gli stem separati (voce, batteria, basso, altro). La voce è il target; la parte strumentale forma "l' accompagnamento".

## Struttura del repository

```
.
├── README.md
├── vocal_separation.ipynb   # Notebook: dati, modello, training, valutazione, inferenza
└── train.py                 # Stesso algoritmo ma come script
```


## Setup — Google Colab (consigliato)

Il training richiede una GPU, quindi il notebook è pensato per Colab.

1. Scarica **MUSDB18** (versione compressa, ~4.4 GB) da [sigsep.github.io/datasets/musdb](https://sigsep.github.io/datasets/musdb.html).
2. Carica lo zip su Google Drive in `AudioSeparation/musdb18.zip`.
3. Apri `vocal_separation.ipynb` su [colab.research.google.com](https://colab.research.google.com).
4. Attiva la GPU: Runtime → Cambia tipo di runtime → Acceleratore hardware → GPU.
5. Se il tuo zip si trova altrove, aggiorna `ZIP_PATH` nella cella 3 del notebook.

Note pratiche:

- Il dataset viene estratto nello storage locale della sessione (`/content/musdb18`), che si azzera quando la sessione si interrompe: va solo riestratto, lo zip su Drive resta.
- Le librerie installate con `pip` vanno reinstallate a ogni nuova sessione (lo fa la cella 2).
- I checkpoint del modello sono salvati su Drive (`AudioSeparation/checkpoints/latest.pt`), quindi sopravvivono ai reset.
- Elaborare brani interi può esaurire la RAM di Colab..

## Setup — Locale con Anaconda

Da evitare il setup in locale se non si disponde di una GPU, ma comunque utile per esplorare i dati o provare lo script.

```bash
# Crea e attiva l'ambiente dedicato
conda create -n audiosep python=3.10 -y
conda activate audiosep

# ffmpeg serve a stempeg per leggere i file .stem.mp4
conda install -c conda-forge ffmpeg -y

# PyTorch (versione CPU; con GPU NVIDIA scegli la build CUDA adatta su pytorch.org)
pip install torch torchvision torchaudio

# Altre librerie
pip install librosa soundfile matplotlib numpy jupyterlab museval stempeg

# Avvia Jupyter
jupyter lab
```

| Azione | Comando |
|---|---|
| Attivare l'ambiente | `conda activate audiosep` |
| Disattivare l'ambiente | `conda deactivate` |
| Elenco ambienti | `conda env list` |
| Rimuovere l'ambiente | `conda env remove -n audiosep` |

In locale, nella cella 3 del notebook imposta `DATASET_ROOT` e `CHECKPOINT_DIR` su cartelle locali.

## Esecuzione

### Notebook

Esegui le celle in ordine, dall'alto:

1. Setup: installazione librerie, mount di Drive, estrazione del dataset.
2. Esplorazione dati: caricamento di un brano, ascolto di mix/voce/accompagnamento, spettrogrammi.
3. Dataset e DataLoader: coppie (spettrogramma del mix, maschera target) calcolate al volo.
4. Modello: definizione della U-Net.
5. Training: ciclo di addestramento con checkpoint ad ogni epoca..
6. Valutazione: tabella SDR di baseline.
7. Inferenza: separazione della voce da un file audio a scelta. Imposta `YOUR_AUDIO_FILE` col percorso del tuo file; viene elaborato un estratto di 30 secondi.

### Script

```bash
python train.py --dataset_root ./musdb18 --checkpoint_dir ./checkpoints --epochs 20
```

Parametri: `--epochs`, `--batch_size`, `--lr`, `--segment_seconds`, `--eval_songs`. Lo script fa training (con ripresa da checkpoint) e poi la valutazione. Le funzioni `separate_file()` e `evaluate_table()` si possono richiamare anche da un altro script.

## Dettagli tecnici

- Rappresentazione: spettrogramma di magnitudine, STFT con `n_fft=2048`, `hop_length=512`, audio mono a 44.1 kHz.
- Segmenti di training: estratti casuali di 4 secondi per brano, ripescati ad ogni passaggio (così lo stesso brano fornisce segmenti diversi) e per contenere la memoria.
- Target: Ideal Ratio Mask, `|voce| / (|voce| + |accompagnamento| + ε)`.
- Modello: U-Net con 4 livelli di encoder/decoder (16→32→64→128 canali, bottleneck a 256), skip connections e uscita con sigmoid.
- Loss: MSE tra maschera predetta e maschera target. Ottimizzatore Adam, learning rate 1e-3, batch size 4, 20 epoche.
- Metrica: SDR (Signal to Distortion Ratio, in dB) calcolato con `museval`.

## Risultati

### Training

La train loss scende da circa 0.124 a circa 0.095, quasi tutto nelle prime 2-3 epoche, poi si stabilizza. La validation loss è molto rumorosa e termina a circa 0.092, vicina alla train loss: non si osserva overfitting, ma il miglioramento si ferma presto.

### Valutazione (SDR)

Per interpretare l'SDR del modello lo si confronta con due riferimenti calcolati sugli stessi estratti:

- Baseline: il mix intero usato come "voce", rappresenta il livello minimo da battere.
- Oracolo: la maschera ideale applicata al mix, rappresenta il massimo ottenibile con questo approccio.

SDR in dB, un estratto di 30 secondi per brano (dal secondo 30), primi 10 brani del test set in ordine alfabetico, valore mediano su finestre di 1 secondo:

| Brano | Baseline | Modello | Oracolo |
|---|---:|---:|---:|
| AM Contra - Heart Peripheral | -4.21 | -1.36 | 13.52 |
| Al James - Schoolboy Facination | -1.57 | 4.05 | 9.66 |
| Angels In Amplifiers - I'm Alright | -2.88 | 3.17 | 10.99 |
| Arise - Run Run Run | -7.61 | -2.16 | 8.83 |
| BKS - Bulldozer | -10.72 | -6.94 | 9.12 |
| BKS - Too Much | -1.67 | 3.71 | 13.03 |
| Ben Carrigan - We'll Talk About It All Tonight | -7.36 | -0.42 | 7.00 |
| Bobby Nobody - Stitch Up | -2.76 | 3.23 | 9.61 |
| Buitraker - Revo X | -10.63 | -8.89 | 8.12 |
| Carlos Gonzalez - A Place For Us | -1.76 | 2.62 | 10.85 |
| **Mediana** | **-3.55** | **1.10** | **9.64** |
| Media | -5.12 | -0.30 | 10.07 |

Il modello supera la baseline in 10 brani su 10. La mediana è circa 4.7 dB sopra la baseline e circa 8.5 dB sotto l'oracolo.

### Note metodologiche

- La valutazione riguarda solo 10 brani su 50  del test set e un solo estratto da 30 secondi per brano, non il brano intero né tutto il test set.
- Su brani interi i valori sono più bassi, perché entrano anche i passaggi senza voce. In una misura preliminare su 2 brani interi (media sulle finestre) si è ottenuto un SDR medio di circa -8.7 dB, quindi i numeri della tabella non vanno letti come prestazione sul brano completo.

## Limiti e possibili miglioramenti

Limiti noti dell'implementazione:

- Modello piccolo e training breve, solo 20 epoche, pensato per girare su Colab gratuito.
- Segmenti corti solo 4 secondi: il modello vede poco contesto musicale.
- Audio mono, quindi la separazione ignora l'informazione stereo.

Possibili miglioramenti:

- Normalizzare l'input, ad esempio con `log(1 + magnitudine)`, data la grande gamma dinamica delle magnitudini.
- Allenare più a lungo, con segmenti più lunghi o un modello più grande.
- Confronto con un modello pre-addestrato per averlo come benchmark.
