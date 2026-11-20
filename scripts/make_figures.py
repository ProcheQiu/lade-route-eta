"""Spatial and temporal maps of the cleaned data -> docs/figures/*.png (run after scripts/clean_data.py)
Usage: python scripts/make_figures.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from lade import config as C
C.FIGS.mkdir(parents=True, exist_ok=True)
dd, fd = str(C.PROC), str(C.FIGS)
D = pd.read_pickle(C.PROC / 'delivery_clean.pkl'); P = pd.read_pickle(C.PROC / 'pickup_clean.pkl')
X0, X1, Y0, Y1 = 121.05, 121.97, 30.70, 31.66
ASP = 1 / np.cos(np.radians(31.2))
BINS = (int((X1 - X0) / 0.004), int((Y1 - Y0) / 0.004))   # ~400 m cells
LAND = {"People's Sq.": (121.4737, 31.2304), 'Lujiazui': (121.5055, 31.2397), 'Hongqiao Apt.': (121.3363, 31.1979),
        'Pudong Apt.': (121.8083, 31.1443), 'Xujiahui': (121.4374, 31.1955), 'Songjiang': (121.2277, 31.0322)}
OFF = {"People's Sq.": (-5, 6), 'Lujiazui': (5, 4), 'Xujiahui': (4, -10), 'Hongqiao Apt.': (-5, 4)}
allx = np.concatenate([D.lng, P.lng, D.finish_gps_lng.dropna(), P.finish_gps_lng.dropna()])
ally = np.concatenate([D.lat, P.lat, D.finish_gps_lat.dropna(), P.finish_gps_lat.dropna()])
BG, _, _ = np.histogram2d(allx, ally, bins=BINS, range=[[X0, X1], [Y0, Y1]])
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})

def base(ax, title=None, landmarks=True, scale=True):
    ax.imshow((BG.T > 0).astype(float), origin='lower', extent=[X0, X1, Y0, Y1], cmap='Greys', vmin=0, vmax=6, aspect=ASP, interpolation='nearest')
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect(ASP)
    ax.tick_params(labelsize=7, length=2)
    if landmarks:
        for k, (x, y) in LAND.items():
            ax.plot(x, y, marker='o', ms=3.5, mfc='white', mec='black', mew=0.8, zorder=5)
            ax.annotate(k, (x, y), xytext=OFF.get(k, (4, 3)), textcoords='offset points', ha='right' if OFF.get(k, (4, 3))[0] < 0 else 'left', fontsize=6.5, zorder=6,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    if scale:
        dx = 10 / (111.32 * np.cos(np.radians(31.2)))
        ax.plot([X1 - 0.04 - dx, X1 - 0.04], [Y0 + 0.03] * 2, color='black', lw=2)
        ax.text(X1 - 0.04 - dx / 2, Y0 + 0.045, '10 km', ha='center', fontsize=7)
    if title: ax.set_title(title, fontsize=9, loc='left')

def dens(ax, x, y, cmap, vmax=None):
    H, _, _ = np.histogram2d(x, y, bins=BINS, range=[[X0, X1], [Y0, Y1]])
    H = np.where(H > 0, H, np.nan)
    return ax.imshow(H.T, origin='lower', extent=[X0, X1, Y0, Y1], cmap=cmap, norm=LogNorm(1, vmax or np.nanmax(H)), aspect=ASP, interpolation='nearest', zorder=3)

# ---- Fig 1: spatial overview ----
fig, axs = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
im = dens(axs[0], D.lng, D.lat, 'Blues'); base(axs[0], f'Delivery stops (LaDe-D), n = {len(D):,}')
fig.colorbar(im, ax=axs[0], shrink=0.6, label='packages per ~400 m cell (log)')
im = dens(axs[1], P.lng, P.lat, 'Oranges'); base(axs[1], f'Pickup stops (LaDe-P), n = {len(P):,}')
fig.colorbar(im, ax=axs[1], shrink=0.6, label='packages per ~400 m cell (log)')
base(axs[2], 'Sampled regions: delivery (blue) vs pickup (orange)')
for df, c in [(D, 'tab:blue'), (P, 'tab:orange')]:
    cen = df.groupby('region_id')[['lng', 'lat']].median()
    s = df.sample(200000, random_state=0)
    axs[2].scatter(s.lng, s.lat, s=0.2, c=c, alpha=0.25, lw=0, zorder=3, rasterized=True)
    axs[2].scatter(cen.lng, cen.lat, s=10, c=c, edgecolors='black', linewidths=0.4, zorder=4)
fig.suptitle('Where LaDe Shanghai packages are: grey = footprint of every recorded stop and GPS fix', fontsize=11, x=0.01, ha='left')
fig.savefig(f'{fd}/fig1_spatial.png', dpi=130); plt.close(fig)

# ---- Fig 2: time of day on the map ----
slots = [(8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 22)]
fig, axs = plt.subplots(2, 6, figsize=(16.5, 6.6), constrained_layout=True)
for r, (df, cmap, lab) in enumerate([(D, 'Blues', 'Delivery'), (P, 'Oranges', 'Pickup')]):
    H_all = [np.histogram2d(df.lng[df.finish_hour.between(a, b - 1)], df.lat[df.finish_hour.between(a, b - 1)], bins=BINS, range=[[X0, X1], [Y0, Y1]])[0].max() for a, b in slots]
    vmax = max(H_all)
    for c, (a, b) in enumerate(slots):
        m = df.finish_hour.between(a, b - 1); ax = axs[r, c]
        im = dens(ax, df.lng[m], df.lat[m], cmap, vmax)
        base(ax, f'{lab} {a:02d}:00–{b:02d}:00  ({m.mean():.0%})', landmarks=(c == 0), scale=(c == 5))
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs[r, :], shrink=0.8, label=f'{lab.lower()} finishes per cell (log)')
fig.suptitle('When packages are finished, by time of day (share of the day\'s finishes in brackets)', fontsize=11, x=0.01, ha='left')
fig.savefig(f'{fd}/fig2_time_of_day.png', dpi=120); plt.close(fig)

# ---- Fig 3: month on the map ----
months = [5, 6, 7, 8, 9, 10]; mname = dict(zip(months, ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']))
fig, axs = plt.subplots(2, 6, figsize=(16.5, 6.6), constrained_layout=True)
for r, (df, cmap, lab) in enumerate([(D, 'Blues', 'Delivery'), (P, 'Oranges', 'Pickup')]):
    vmax = max(np.histogram2d(df.lng[df.month == mo], df.lat[df.month == mo], bins=BINS, range=[[X0, X1], [Y0, Y1]])[0].max() for mo in months)
    for c, mo in enumerate(months):
        m = df.month == mo; ax = axs[r, c]
        im = dens(ax, df.lng[m], df.lat[m], cmap, vmax)
        base(ax, f'{lab}, {mname[mo]}  (n = {m.sum():,})', landmarks=(c == 0), scale=(c == 5))
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs[r, :], shrink=0.8, label=f'{lab.lower()} packages per cell (log)')
fig.suptitle('How coverage changes month by month (May is a partial month)', fontsize=11, x=0.01, ha='left')
fig.savefig(f'{fd}/fig3_by_month.png', dpi=120); plt.close(fig)

# ---- Fig 4: what varies by place ----
fig, axs = plt.subplots(1, 2, figsize=(11, 5.6), constrained_layout=True)
G = 0.012
def cellstat(df, val, fn, minn):
    gx = ((df.lng - X0) / G).astype(int); gy = ((df.lat - Y0) / G).astype(int)
    st = df.assign(gx=gx, gy=gy).groupby(['gx', 'gy'])[val].agg([fn, 'size'])
    st = st[st['size'] >= minn]
    M = np.full((int((X1 - X0) / G) + 1, int((Y1 - Y0) / G) + 1), np.nan)
    M[st.index.get_level_values(0), st.index.get_level_values(1)] = st[fn].values
    return M
M = cellstat(D, 'dur_min', 'median', 30)
im = axs[0].imshow(M.T, origin='lower', extent=[X0, X0 + G * M.shape[0], Y0, Y0 + G * M.shape[1]], cmap='viridis', vmin=30, vmax=180, aspect=ASP, zorder=3)
base(axs[0], 'Delivery: median minutes from accept to delivered (~1.2 km cells, n >= 30)')
fig.colorbar(im, ax=axs[0], shrink=0.7, label='minutes')
M = cellstat(P.assign(late=P.late.astype(float) * 100), 'late', 'mean', 30)
im = axs[1].imshow(M.T, origin='lower', extent=[X0, X0 + G * M.shape[0], Y0, Y0 + G * M.shape[1]], cmap='Reds', vmin=0, vmax=10, aspect=ASP, zorder=3)
base(axs[1], 'Pickup: % picked up after the time window ends (cells, n >= 30)')
fig.colorbar(im, ax=axs[1], shrink=0.7, label='% late (capped at 10)')
fig.savefig(f'{fd}/fig4_service_by_place.png', dpi=130); plt.close(fig)

# ---- aggregates for doc charts ----
h = pd.concat([D.finish_hour.value_counts(normalize=True).rename('Delivery'), P.finish_hour.value_counts(normalize=True).rename('Pickup')], axis=1).fillna(0).sort_index().loc[7:22]
rows = [{'hour': int(k), 'series': s, 'share': round(float(h.loc[k, s]) * 100, 2)} for k in h.index for s in ['Delivery', 'Pickup']]
pd.DataFrame(rows).to_csv(f'{dd}/hourly_share.csv', index=False)
dv = pd.concat([D.groupby('ds_date').size().rename('Delivery'), P.groupby('ds_date').size().rename('Pickup')], axis=1).fillna(0).astype(int)
rows = [{'date': d.strftime('%m-%d'), 'series': s, 'packages': int(dv.loc[d, s])} for d in dv.index for s in ['Delivery', 'Pickup']]
pd.DataFrame(rows).to_csv(f'{dd}/daily_volume.csv', index=False)
print(h.round(3).T.to_string()); print(dv.describe().round(0))
print('weekly autocorr delivery', pd.Series(dv.Delivery.values).autocorr(7).round(2), 'pickup', pd.Series(dv.Pickup.values).autocorr(7).round(2))

# ---- Fig 5 / 6: hourly profile and daily volume (the two charts in the design doc) ----
fig, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
ax.plot(h.index, h.Delivery * 100, marker='o', color='tab:blue', label='Delivery')
ax.plot(h.index, h.Pickup * 100, marker='o', color='tab:orange', label='Pickup')
ax.axvline(12, ls='--', color='grey', lw=1); ax.set_ylabel('% of finishes'); ax.set_xlabel('hour of day'); ax.legend()
ax.set_title('Pickups bunch in the morning; deliveries run into the evening', loc='left', fontsize=10)
fig.savefig(f'{fd}/fig5_hourly.png', dpi=130); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
for c, col in [('Delivery', 'tab:blue'), ('Pickup', 'tab:orange')]:
    ax.plot(dv.index, dv[c], color=col, alpha=0.3, lw=0.8); ax.plot(dv.index, dv[c].rolling(7, min_periods=1).mean(), color=col, lw=2, label=c)
for d in ['2022-08-18', '2022-09-24']:
    ax.axvline(pd.Timestamp(d), ls='--', color='grey', lw=1)
ax.set_ylabel('packages per day'); ax.legend(); ax.set_title('Daily volume (thin) and 7-day average (thick); dashed = delivery train/val/test cuts', loc='left', fontsize=10)
fig.savefig(f'{fd}/fig6_daily_volume.png', dpi=130); plt.close(fig)
