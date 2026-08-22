"""
Offline Online-Inference Simulation（真實資料版）
================================================
用歷史 .pt 資料模擬 online 流程：
  1. 從 base model 開始
  2. 逐首歌做 trial-level inference → 計算正確率
  3. 每首歌結束後執行 Calib（RL open）+ RL Loop reward 更新
  4. 只保留最新模型，刪除舊 checkpoint
  5. 輸出 per-song accuracy 圖表

用法:
    cd d:\\2026_BCI_-
    python plot_acc_vs_time.py
    python plot_acc_vs_time.py --ft-epochs 10
    python plot_acc_vs_time.py --accounts 0521_tsai_2
"""
import os, sys, glob, argparse, random, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from datetime import datetime

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# ── 路徑設定 ──
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.join(SCRIPT_DIR, "combine_all_new")
sys.path.insert(0, PROJECT_DIR)

import config
import global_state
from eeg.eeg_model import EEGNet_our_ver, load_EEGNet_our_ver_params
from eeg.braindecode_trainer import BraindecodeTrainer
import eeg.preprocess as pps
from rl.bandit import (NeuralContextualBandit, BanditAction, ContextIDX,
                       UpdateScheduler, ContextNormalizer)
from rl.rl_interface import RL_interface

# ─── 常數 ───
SEED         = 42
SEGMENT_LEN  = 500
N_CHANS      = config.CHOOSE_CHANNELS
N_CLASSES    = config.N_Class
TRAIN_STRIDE = 50
VALID_STRIDE = 100

BASE_MODEL   = os.path.join(PROJECT_DIR, "eeg", "base_model", "pretrain_DL_4class.pth")
BANDIT_PATH  = os.path.join(PROJECT_DIR, "rl", "rl_pretrain", "pretrained_bandit.pt")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "acc_vs_time_plots")

ACCOUNTS = [
    "0514_hsu_noRL_loop",
    "0521_tsai_2",
]

ACCOUNT_COLORS = {
    "0514_hsu_noRL_loop": "#2ECC71",
    "0521_tsai_2":        "#9B59B6",
}
ACCOUNT_MARKERS = {
    "0514_hsu_noRL_loop": "D",
    "0521_tsai_2": "v",
}

SONG_COLORS = [
    "#4A90D9", "#E74C3C", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#3498DB",
    "#E91E63", "#00BCD4",
]


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================================================
# Model helpers
# ================================================================
def build_model(device):
    m = EEGNet_our_ver(n_chans=N_CHANS, n_outputs=N_CLASSES, n_times=SEGMENT_LEN).to(device)
    return m

def load_ckpt(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def infer_trials(model, x_data, y_data, device):
    """對每個 trial 取最後 500 samples → demean+bandpass → 推論，回傳 accuracy"""
    model.eval()
    correct = 0
    for i in range(len(x_data)):
        seg = x_data[i][:, -SEGMENT_LEN:]
        seg = pps.each_channel_demean(seg, axis=1)
        seg = pps.bandpass(seg, axis=-1)
        xb = torch.from_numpy(seg.copy()).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(torch.argmax(model(xb)).cpu().item())
        if pred == y_data[i]:
            correct += 1
    return correct / max(len(y_data), 1)


def sliding_window(trial_T, stride, seg_len=SEGMENT_LEN):
    """trial_T: (T, C)"""
    wins = []
    for s in range(0, trial_T.shape[0] - seg_len + 1, stride):
        w = trial_T[s:s+seg_len]
        w = pps.each_channel_demean(w, axis=0)
        w = pps.bandpass(w)
        wins.append(w)
    return wins

def to_dataset(segs, labs):
    if not segs: return None
    dx = np.transpose(np.stack(segs), (0, 2, 1))
    X = torch.tensor(dx).unsqueeze(1).float()
    y = F.one_hot(torch.tensor(labs).long(), num_classes=N_CLASSES).float()
    return TensorDataset(X, y)


# ================================================================
# Calib + RL (one song)
# ================================================================
def calib_update(model, cum_x, cum_y, rl_if, device, tmp_dir, ft_epochs, last_val_loss, use_rl=True):
    """
    Calib 模式：用累積的所有資料做 fine-tune。
    use_rl=True 則 RL 決定 stride，否則固定為 TRAIN_STRIDE。
    回傳 (model, val_acc, val_loss, session_records)
    """
    indices = np.arange(len(cum_x))
    try:
        tr_idx, va_idx = train_test_split(indices, test_size=0.25, stratify=cum_y, random_state=SEED)
    except ValueError:
        tr_idx, va_idx = train_test_split(indices, test_size=0.25, random_state=SEED)

    session_records = []
    tr_segs, tr_labs = [], []
    va_segs, va_labs = [], []

    for ti in tr_idx:
        trial_T = cum_x[ti].T
        stride = TRAIN_STRIDE
        if use_rl:
            hit = {"eeg": cum_x[ti], "probs": np.ones(N_CLASSES)/N_CLASSES}
            raw_ctx = rl_if.extract_per_hit_context(hit)
            norm_ctx = rl_if.normalizer.update_and_normalize(raw_ctx)
            action = rl_if.bandit.select_action(norm_ctx)
            session_records.append({"context": norm_ctx, "action": action})
            if action == BanditAction.HIGH_WEIGHT: stride = max(1, TRAIN_STRIDE // 2)
            elif action == BanditAction.LOW_WEIGHT: stride = TRAIN_STRIDE * 2
        ws = sliding_window(trial_T, stride)
        if ws:
            tr_segs.extend(ws); tr_labs.extend([cum_y[ti]] * len(ws))

    for vi in va_idx:
        ws = sliding_window(cum_x[vi].T, VALID_STRIDE)
        if ws:
            va_segs.extend(ws); va_labs.extend([cum_y[vi]] * len(ws))

    tr_ds = to_dataset(tr_segs, tr_labs)
    va_ds = to_dataset(va_segs, va_labs)
    if tr_ds is None or va_ds is None or len(tr_ds) < 4 or len(va_ds) < 4:
        return model, None, last_val_loss, session_records

    # 清理舊 tmp
    prefix = "rl_" if use_rl else "pure_"
    model_tmp_dir = os.path.join(tmp_dir, prefix)
    if os.path.isdir(model_tmp_dir): shutil.rmtree(model_tmp_dir, ignore_errors=True)
    os.makedirs(model_tmp_dir, exist_ok=True)
    global_state.EEG_CHECKPOINT_TMP_BASE_DIR = model_tmp_dir

    mk = {"n_chans": tr_ds[0][0].shape[1], "n_outputs": tr_ds[0][1].shape[0], "n_times": tr_ds[0][0].shape[2]}
    trainer = BraindecodeTrainer(EEGNet_our_ver, mk, tr_ds, va_ds, 8, ft_epochs, 1e-5, True, None)
    trainer.model.load_state_dict(model.state_dict())
    trainer.model.to(device)
    hist = trainer.train(freeze_layer=False, patience=10)
    best_ep = int(np.argmin(hist["val_loss"]))
    va_acc = hist["val_acc"][best_ep]
    va_loss = hist["val_loss"][best_ep]

    best_path = os.path.join(model_tmp_dir, f"ft-epoch{best_ep}.pth")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(trainer.model.state_dict())
    model.eval()

    # 只保留最新模型：刪除所有 epoch pth
    for f in glob.glob(os.path.join(model_tmp_dir, "*.pth")):
        try: os.remove(f)
        except: pass

    return model, va_acc, va_loss, session_records


# ================================================================
# RL Loop reward update
# ================================================================
def rl_loop_reward_update(rl_if, session_records, delta_loss):
    """Calib 後回饋 reward 給 bandit"""
    for rec in session_records:
        reward = rl_if.bandit.compute_reward(
            deltaLoss=delta_loss,
            meanEntropy=float(rec["context"][ContextIDX.MEAN_ENTROPY]),
            alphaNorm=float(rec["context"][ContextIDX.ALPHA_POWER_NORM]))
        rl_if.bandit.update(rec["context"], rec["action"], reward)


# ================================================================
# Plot
# ================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Segoe UI'],
    'font.size': 11,
    'axes.facecolor': '#FAFBFD',
    'figure.facecolor': '#FFFFFF',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

def plot_account(account, song_names, accs_base, accs_pure, accs_rl, output_dir):
    n = len(song_names)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(10, n * 2.2), 7))
    ax.set_facecolor('#F8F9FC')

    ax.plot(x, [a*100 for a in accs_rl], color='#E85D75', lw=2.5, marker='s', ms=7,
            label='Calib+RL (cumulative)', zorder=5)
    ax.plot(x, [a*100 for a in accs_pure], color='#2ECC71', lw=2.5, marker='^', ms=7,
            label='Pure DL (cumulative)', zorder=5)
    ax.plot(x, [a*100 for a in accs_base], color='#4A90D9', lw=2, marker='o', ms=6,
            alpha=0.7, label='Base Model (fixed)', zorder=4)
    ax.axhline(y=25, color='gray', ls=':', alpha=0.6, lw=1.5, label='Chance (25%)')

    for xp, acc in zip(x, accs_rl):
        ax.text(xp, (acc*100) + 1.5,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2C3E50')

    avg_b = np.mean(accs_base)*100; avg_p = np.mean(accs_pure)*100; avg_r = np.mean(accs_rl)*100
    ax.set_xticks(x)
    ax.set_xticklabels(song_names, fontsize=9, ha='center')
    ax.set_xlabel('Song', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Online Sim — {account}\nBase: {avg_b:.1f}% | Pure DL: {avg_p:.1f}% | Calib+RL: {avg_r:.1f}%',
                 fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_ylim(0, max(max(accs_rl), max(accs_pure), max(accs_base))*100 + 15)
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.legend(fontsize=10, loc='upper left')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax.spines[sp].set_color('#CCCCCC')
    plt.tight_layout()
    p = os.path.join(output_dir, f"online_sim_{account}.png")
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  → Saved: {p}")
    return p


def plot_combined(all_data, output_dir):
    """all_data: dict of account -> (song_names, accs_base, accs_pure, accs_rl)"""
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_facecolor('#F8F9FC')

    all_labels, all_accs_b, all_accs_p, all_accs_r, all_colors_bar = [], [], [], [], []
    x_pos, xi = [], 0
    group_ranges = {}

    for acct, (names, ab, ap, ar) in all_data.items():
        color = ACCOUNT_COLORS.get(acct, '#888')
        start = xi
        for i in range(len(names)):
            all_labels.append(names[i])
            all_accs_b.append(ab[i]*100)
            all_accs_p.append(ap[i]*100)
            all_accs_r.append(ar[i]*100)
            all_colors_bar.append(color)
            x_pos.append(xi); xi += 1
        group_ranges[acct] = (start, xi-1)
        xi += 1

    x_pos = np.array(x_pos)
    for xp, av in zip(x_pos, all_accs_r):
        ax.text(xp, av+1.5, f'{av:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold', color='#2C3E50')

    xi = 0
    for acct, (names, ab, ap, ar) in all_data.items():
        if not names: continue
        color = ACCOUNT_COLORS.get(acct, '#888')
        mkr = ACCOUNT_MARKERS.get(acct, 'o')
        n = len(names)
        xv = list(range(xi, xi+n))
        
        # Base Model 線 (灰色虛線)
        ax.plot(xv, [a*100 for a in ab], color='gray', lw=1.5, ls='--', alpha=0.7,
                marker='o', ms=5, markerfacecolor='white', markeredgecolor='gray', markeredgewidth=1.0,
                zorder=4)
                
        # Pure DL 線 (綠色)
        ax.plot(xv, [a*100 for a in ap], color='#2ECC71', lw=2.0, alpha=0.8,
                marker='^', ms=6, markerfacecolor='#2ECC71', markeredgecolor='white', markeredgewidth=1.0,
                zorder=5)
                
        # Calib+RL 線 (依帳號上色)
        ax.plot(xv, [a*100 for a in ar], color=color, lw=2.5, alpha=0.9,
                marker=mkr, ms=8, markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.5,
                label=acct, zorder=6)
        xi += n + 1

    ax.axhline(y=25, color='#E74C3C', ls=':', alpha=0.6, lw=1.5, label='Chance (25%)')
    
    # 建立手動的 Legend 項目以包含 Base Model 和 Pure DL
    import matplotlib.lines as mlines
    base_legend = mlines.Line2D([], [], color='gray', lw=1.5, ls='--', marker='o', ms=5,
                                markerfacecolor='white', markeredgecolor='gray', label='Base Model (fixed)')
    pure_legend = mlines.Line2D([], [], color='#2ECC71', lw=2.0, marker='^', ms=6,
                                markerfacecolor='#2ECC71', markeredgecolor='white', label='Pure DL (cumulative)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_labels, fontsize=7, ha='center')

    xi = 0
    for acct, (names, _, _, _) in all_data.items():
        if not names: continue
        n = len(names)
        mid = xi + (n-1)/2
        ax.annotate(acct, xy=(mid, 0), xycoords=('data','axes fraction'),
                    xytext=(0,-40), textcoords='offset points',
                    ha='center', va='top', fontsize=9, fontweight='bold',
                    color=ACCOUNT_COLORS.get(acct, '#888'))
        xi += n + 1

    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Online Sim — All Accounts (Calib + RL Loop)', fontsize=16, fontweight='bold', color='#2C3E50')
    if all_accs_r:
        ax.set_ylim(0, max(all_accs_r) + 15)
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    handles, labels = ax.get_legend_handles_labels()
    handles.append(base_legend)
    handles.append(pure_legend)
    ax.legend(handles=handles, fontsize=9, loc='upper right', ncol=2)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax.spines[sp].set_color('#CCCCCC')
    plt.tight_layout(); plt.subplots_adjust(bottom=0.18)
    p = os.path.join(output_dir, "online_sim_ALL_ACCOUNTS.png")
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"\n  → Combined plot saved: {p}")


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Offline Online Simulation (real data)")
    parser.add_argument("--ft-epochs", type=int, default=5, help="Fine-tune epochs per song")
    parser.add_argument("--accounts", nargs="*", default=None, help="Specify accounts (default: all)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    accounts = args.accounts or ACCOUNTS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Sim] Device: {device}")
    print(f"[Sim] FT epochs: {args.ft_epochs}")
    print(f"[Sim] Accounts: {accounts}")

    account_base = os.path.join(PROJECT_DIR, "account")
    tmp_dir = os.path.join(OUTPUT_DIR, "_ft_tmp")
    all_plot_data = {}

    for account in accounts:
        rt_dir = os.path.join(account_base, account, "real_time_data")
        pt_files = sorted(glob.glob(os.path.join(rt_dir, "*.pt"))) if os.path.isdir(rt_dir) else []
        if not pt_files:
            print(f"\n[SKIP] {account}: no .pt files")
            continue

        print(f"\n{'='*60}")
        print(f"  Account: {account}  ({len(pt_files)} songs)")
        print(f"{'='*60}")

        # 載入 base model
        base_model = load_ckpt(build_model(device), BASE_MODEL, device)
        pure_dl_model = load_ckpt(build_model(device), BASE_MODEL, device)
        adaptive_model = load_ckpt(build_model(device), BASE_MODEL, device)

        # 載入 RL
        rl_if = RL_interface(bandit=NeuralContextualBandit(epsilon=0.5, lambdaCost=0.1))
        if os.path.exists(BANDIT_PATH):
            rl_if.bandit.load(BANDIT_PATH)

        song_names, accs_base, accs_pure, accs_rl = [], [], [], []
        cum_x, cum_y = [], []
        last_val_loss_rl = float("inf")
        last_val_loss_pure = float("inf")

        for si, pt_path in enumerate(pt_files):
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            x = data["x_data"].numpy()
            y = data["y_data"].numpy()
            fname = os.path.basename(pt_path).replace(".pt","")
            time_part = fname.split("_")[-1]
            sname = f"Song{si+1}\n({time_part})"

            # ── Inference (base) ──
            acc_b = infer_trials(base_model, x, y, device)
            # ── Inference (pure DL) ──
            acc_p = infer_trials(pure_dl_model, x, y, device)
            # ── Inference (adaptive) ──
            acc_r = infer_trials(adaptive_model, x, y, device)

            song_names.append(sname)
            accs_base.append(acc_b)
            accs_pure.append(acc_p)
            accs_rl.append(acc_r)

            print(f"\n  Song [{si+1}/{len(pt_files)}] {fname}")
            print(f"    trials={len(y)}  Base={acc_b*100:.1f}%  PureDL={acc_p*100:.1f}%  Calib+RL={acc_r*100:.1f}%")

            # ── Data Accumulation ──
            cum_x.append(x); cum_y.append(y)
            all_x = np.concatenate(cum_x, axis=0)
            all_y = np.concatenate(cum_y, axis=0)

            # ── Calib (Pure DL) ──
            pure_dl_model, p_va_acc, p_va_loss, _ = calib_update(
                pure_dl_model, all_x, all_y, rl_if, device, tmp_dir, args.ft_epochs, last_val_loss_pure, use_rl=False)
            if p_va_acc is not None:
                last_val_loss_pure = p_va_loss
                print(f"    PureDL FT: val_acc={p_va_acc*100:.1f}%  val_loss={p_va_loss:.4f}")

            # ── Calib + RL ──
            adaptive_model, r_va_acc, r_va_loss, sess_recs = calib_update(
                adaptive_model, all_x, all_y, rl_if, device, tmp_dir, args.ft_epochs, last_val_loss_rl, use_rl=True)

            if r_va_acc is not None:
                delta_loss = r_va_loss - last_val_loss_rl if last_val_loss_rl != float("inf") else 0.0
                last_val_loss_rl = r_va_loss
                print(f"    RL FT: val_acc={r_va_acc*100:.1f}%  val_loss={r_va_loss:.4f}  Δloss={delta_loss:.4f}")
                # RL Loop reward
                rl_loop_reward_update(rl_if, sess_recs, delta_loss)
            else:
                print(f"    RL FT: skipped (insufficient data)")

        # ── Plot per account ──
        if song_names:
            all_plot_data[account] = (song_names, accs_base, accs_pure, accs_rl)
            plot_account(account, song_names, accs_base, accs_pure, accs_rl, OUTPUT_DIR)

    # ── Combined plot ──
    if all_plot_data:
        plot_combined(all_plot_data, OUTPUT_DIR)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"{'Account':<25} {'Song':<15} {'Base':>8} {'PureDL':>8} {'Calib+RL':>10}")
    print("-"*72)
    for acct, (names, ab, ap, ar) in all_plot_data.items():
        for i in range(len(names)):
            lbl = names[i].replace('\n',' ')
            print(f"{acct:<25} {lbl:<15} {ab[i]*100:>7.1f}% {ap[i]*100:>7.1f}% {ar[i]*100:>9.1f}%")
        avg_b = np.mean(ab)*100; avg_p = np.mean(ap)*100; avg_r = np.mean(ar)*100
        print(f"{'  '+acct+' avg':<25} {'':15} {avg_b:>7.1f}% {avg_p:>7.1f}% {avg_r:>9.1f}%")

    # 清理 tmp
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n[DONE] All plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
