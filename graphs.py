import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

BLUE1 = '#1A4F8A'
BLUE2 = '#4A90C4'
BLUE3 = '#7FB3D9'
GREY1 = '#5A6272'
GREY2 = '#A8B4C0'
BG    = '#F8FBFF'
RED   = '#C0392B'

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.edgecolor':    GREY2,
    'figure.dpi':        150,
})

FOLDER = os.path.dirname(os.path.abspath(__file__))

df   = pd.read_csv(os.path.join(FOLDER, 'data.csv'))
coef = pd.read_csv(os.path.join(FOLDER, 'model_coefficients.csv'))

FIRMS = ['Lukoil','Tatneft','NOVATEK','ROSNEFT','GAZPROM',
         'ROSSETI','RusHydro','InterRAO',
         'ALROSA','POLYMETAL','NORNICKEL','RUSAL','NLMK','SEVERSTAL']

SECTOR_COLORS = {
    'нефтезаговая': BLUE1,
    'энергетика':   BLUE2,
    'металлургия':  GREY1,
}


def plot_missing():
    raw_cols = ['invsoc_rev','energy_rev','atm_rev','LTIFR','Employee_turnover',
                'assets','ROA_raw','Leverage_raw','ln_tobinq_raw','ev_ebitda','pbv']
    raw_lbls = ['СоцИнв/Выр','Энерг/Выр','АтмВыбр/Выр','LTIFR','Текучесть',
                'Активы','ROA','ФинРычаг','ln(TobinQ)','EV/EBITDA','P/BV']

    miss = np.zeros((len(FIRMS), len(raw_cols)), dtype=int)
    for i, firm in enumerate(FIRMS):
        sub = df[df['firm'] == firm]
        for j, col in enumerate(raw_cols):
            miss[i, j] = sub[col].isnull().sum() if col in sub.columns else 10

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(BG)

    cmap = LinearSegmentedColormap.from_list('miss', [BG, BLUE2, RED])
    im = ax.imshow(miss.T, aspect='auto', cmap=cmap, vmin=0, vmax=10)

    for i in range(len(FIRMS)):
        for j in range(len(raw_cols)):
            v = miss[i, j]
            ax.text(i, j, str(v) if v > 0 else '·',
                    ha='center', va='center', fontsize=7.5,
                    color='white' if v > 5 else GREY1,
                    fontweight='bold' if v > 0 else 'normal')

    ax.set_xticks(range(len(FIRMS)))
    ax.set_xticklabels(FIRMS, rotation=45, ha='right', fontsize=7.5)
    ax.set_yticks(range(len(raw_cols)))
    ax.set_yticklabels(raw_lbls, fontsize=8)
    ax.set_title('Карта пропущенных значений (число пропусков из 10 наблюдений)',
                 pad=8, color=BLUE1, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Кол-во пропусков', fontsize=7.5, color=GREY1)

    plt.tight_layout()
    fig.savefig(os.path.join(FOLDER, '01_missing_map.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_boxplots():
    raw_cols = ['invsoc_rev','energy_rev','atm_rev','LTIFR','Employee_turnover']
    log_cols = ['ln_invsoc_rev','ln_energy_rev','ln_atm_rev','ln_LTIFR','ln_turnover']
    raw_lbls = ['СоцИнв/Выр','Энерг/Выр','АтмВыбр/Выр','LTIFR','Текучесть']
    log_lbls = ['ln(СоцИнв/Выр)','ln(Энерг/Выр)','ln(АтмВыбр/Выр)','ln(LTIFR)','ln(Текучесть)']

    fig, axes = plt.subplots(2, 5, figsize=(14, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Выбросы в ESG-переменных: до (исходные) и после логарифмирования с вин. 1–99%',
                 fontsize=10, fontweight='bold', color=BLUE1, y=1.01)

    box_props = dict(
        patch_artist=True,
        medianprops=dict(color=RED, lw=2),
        whiskerprops=dict(color=BLUE1),
        capprops=dict(color=BLUE1),
        flierprops=dict(marker='o', color=RED, markersize=3, alpha=0.5, linestyle='none')
    )

    for i, (rc, lc, rl, ll) in enumerate(zip(raw_cols, log_cols, raw_lbls, log_lbls)):
        for row, (col, lbl, fill) in enumerate([(rc, rl, BLUE3), (lc, ll, BLUE2)]):
            ax = axes[row, i]
            ax.set_facecolor(BG)
            ax.boxplot(df[col].dropna().values, widths=0.5,
                       boxprops=dict(facecolor=fill, color=BLUE1), **box_props)
            ax.set_title(lbl, fontsize=8,
                         color=BLUE1 if row == 1 else GREY1,
                         fontweight='bold' if row == 1 else 'normal')
            ax.set_xticks([])
            ax.tick_params(labelsize=7)

    axes[0, 0].set_ylabel('Исходные значения', fontsize=8, color=GREY1)
    axes[1, 0].set_ylabel('Логарифм. значения', fontsize=8, color=BLUE1)

    plt.tight_layout()
    fig.savefig(os.path.join(FOLDER, '02_boxplots.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_distributions():
    pairs = [
        ('ln_tobinq_raw',    'ln_tobinq',      'TobinQ',           'ln(TobinQ)'),
        ('invsoc_rev',       'ln_invsoc_rev',   'СоцИнв/Выр',       'ln(СоцИнв/Выр)'),
        ('energy_rev',       'ln_energy_rev',   'Энерг/Выр',        'ln(Энерг/Выр)'),
        ('atm_rev',          'ln_atm_rev',      'АтмВыбр/Выр',      'ln(АтмВыбр/Выр)'),
        ('LTIFR',            'ln_LTIFR',        'LTIFR',            'ln(LTIFR)'),
        ('Employee_turnover','ln_turnover',     'Текучесть кадров', 'ln(Текучесть)'),
    ]

    fig, axes = plt.subplots(len(pairs), 2, figsize=(10, 3.2 * len(pairs)))
    fig.patch.set_facecolor('white')

    for row, (rc, lc, rl, ll) in enumerate(pairs):
        for col, (colname, title) in enumerate([(rc, rl), (lc, ll)]):
            ax = axes[row, col]
            ax.set_facecolor(BG)
            vals = df[colname].dropna().values

            ax.hist(vals, bins=20,
                    color=BLUE3 if col == 0 else BLUE2,
                    edgecolor=BLUE1, linewidth=0.4, alpha=0.9)
            ax.axvline(vals.mean(), color=BLUE1, lw=1.3, ls='--')

            std = vals.std()
            skew = float(np.mean(((vals - vals.mean()) / std) ** 3)) if std > 0 else 0
            kurt = float(np.mean(((vals - vals.mean()) / std) ** 4)) - 3 if std > 0 else 0

            ax.text(0.97, 0.95, f'Асимм={skew:.2f}\nЭксц={kurt:.2f}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=8, color=GREY1, linespacing=1.5)
            ax.set_title(title, fontsize=10,
                         color=BLUE1 if col == 1 else GREY1,
                         fontweight='bold' if col == 1 else 'normal', pad=4)
            ax.set_ylabel('Частота', fontsize=8.5)
            ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FOLDER, '03_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_correlation():
    cols = ['ln_tobinq','ln_invsoc_rev','ln_energy_rev','ln_atm_rev',
            'ln_LTIFR','ln_turnover','ROA','Leverage','ln_assets','gdp_growth','ln_oil']
    lbls = ['ln(TobinQ)','ln(СоцИнв/Выр)','ln(Энерг/Выр)','ln(АтмВыбр/Выр)',
            'ln(LTIFR)','ln(Текучесть)','ROA','ФинРычаг','ln(Активы)','ВВП,%','ln(Нефть)']

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.patch.set_facecolor('white')

    cmap = LinearSegmentedColormap.from_list('corr', [BLUE1, 'white', '#8B1A1A'])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=7.5, color='white' if abs(v) > 0.55 else GREY1)

    ax.set_xticks(range(len(lbls)))
    ax.set_xticklabels(lbls, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(lbls)))
    ax.set_yticklabels(lbls, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02).ax.tick_params(labelsize=7.5)
    ax.set_title('Матрица парных корреляций Пирсона', pad=8, color=BLUE1, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(FOLDER, '04_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_within_scatter():
    esg_cols = ['ln_invsoc_rev','ln_energy_rev','ln_atm_rev','ln_LTIFR','ln_turnover']
    esg_lbls = ['ln(СоцИнв/Выр)','ln(Энерг/Выр)','ln(АтмВыбр/Выр)','ln(LTIFR)','ln(Текучесть)']

    dfw = df[['firm','sector','ln_tobinq'] + esg_cols].dropna().copy()
    for col in ['ln_tobinq'] + esg_cols:
        dfw[col + '_w'] = dfw[col] - dfw.groupby('firm')[col].transform('mean')

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Within-графики: ESG-переменные vs ln(TobinQ) (дисперсия внутри компаний)',
                 fontsize=10, fontweight='bold', color=BLUE1, y=1.01)

    for idx, (ev, lbl) in enumerate(zip(esg_cols, esg_lbls)):
        ax = axes.flatten()[idx]
        ax.set_facecolor(BG)

        for firm in FIRMS:
            sub = dfw[dfw['firm'] == firm]
            if sub.empty:
                continue
            ax.scatter(sub[ev + '_w'], sub['ln_tobinq_w'],
                       color=SECTOR_COLORS.get(sub['sector'].iloc[0], GREY2),
                       alpha=0.6, s=18, zorder=3)

        xv = dfw[ev + '_w'].values
        yv = dfw['ln_tobinq_w'].values
        mask = ~(np.isnan(xv) | np.isnan(yv))
        if mask.sum() > 5:
            b = np.polyfit(xv[mask], yv[mask], 1)
            xl = np.linspace(xv[mask].min(), xv[mask].max(), 50)
            ax.plot(xl, np.polyval(b, xl), color=RED, lw=1.5, ls='--', zorder=4)

        ax.axhline(0, color=GREY2, lw=0.5, ls=':')
        ax.axvline(0, color=GREY2, lw=0.5, ls=':')
        ax.set_xlabel(lbl, fontsize=8)
        ax.set_ylabel('ln(TobinQ) — within' if idx % 3 == 0 else '', fontsize=8)
        ax.set_title(f'{lbl} vs ln(TobinQ)', fontsize=8.5, color=BLUE1)
        ax.tick_params(labelsize=7)

    legend_ax = axes.flatten()[5]
    legend_ax.axis('off')
    legend_ax.legend(
        handles=[mpatches.Patch(color=c, label=s) for s, c in SECTOR_COLORS.items()],
        loc='center', fontsize=9, title='Отрасль', title_fontsize=9, framealpha=0.0
    )

    plt.tight_layout()
    fig.savefig(os.path.join(FOLDER, '05_within_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_coef_lagged():
    lbls = coef['label'].tolist()[:5]
    y    = np.arange(5)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(BG)

    ax.errorbar(coef['fe1_b'][:5],  y - 0.12, xerr=1.96 * coef['fe1_se'][:5],
                fmt='o', color=BLUE1, markersize=6, capsize=4, lw=1.3,
                label='FE one-way (без лага)')
    ax.errorbar(coef['fe1l_b'][:5], y + 0.12, xerr=1.96 * coef['fe1l_se'][:5],
                fmt='s', color=BLUE3, markersize=6, capsize=4, lw=1.3,
                label='FE one-way (ESG с лагом 1 год)')

    ax.axvline(0, color=GREY1, lw=1, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels(lbls, fontsize=9)
    ax.set_xlabel('Оценка коэффициента (95% ДИ)', fontsize=9)
    ax.set_title('ESG-коэффициенты: современные vs лаговые значения (FE one-way)',
                 color=BLUE1, fontweight='bold', pad=8)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(axis='x', color=GREY2, linewidth=0.4, ls=':')

    plt.tight_layout()
    fig.savefig(os.path.join(FOLDER, '06_coef_lagged.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_coef_altdv():
    lbls = coef['label'].tolist()[:5]
    y    = np.arange(5)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor(BG)

    ax.errorbar(coef['fe1_b'][:5],   y - 0.2, xerr=1.96 * coef['fe1_se'][:5],
                fmt='o', color=BLUE1, markersize=6, capsize=4, lw=1.3,
                label='FE: ln(TobinQ)')
    ax.errorbar(coef['fe1ev_b'][:5], y,       xerr=1.96 * coef['fe1ev_se'][:5],
                fmt='s', color=BLUE2, markersize=6, capsize=4, lw=1.3,
                label='FE: ln(EV/EBITDA)')
    ax.errorbar(coef['fe1pb_b'][:5], y + 0.2, xerr=1.96 * coef['fe1pb_se'][:5],
                fmt='^', color=GREY1, markersize=6, capsize=4, lw=1.3,
                label='FE: ln(P/BV)')

    ax.axvline(0, color=GREY1, lw=1, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels(lbls, fontsize=9)
    ax.set_xlabel('Оценка коэффициента (95% ДИ)', fontsize=9)
    ax.set_title('Проверка устойчивости: альтернативные зависимые переменные',
                 color=BLUE1, fontweight='bold', pad=8)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(axis='x', color=GREY2, linewidth=0.4, ls=':')

    plt.tight_layout()
    fig.savefig(os.path.join(FOLDER, '07_coef_altdv.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    plot_missing()
    plot_boxplots()
    plot_distributions()
    plot_correlation()
    plot_within_scatter()
    plot_coef_lagged()
    plot_coef_altdv()
