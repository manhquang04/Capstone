"""Frozen-scale frontier rendering from fully audited figure rows only.

No measurement reads, estimator selection, summary calculation or CSV writes.
The caller constructs and independently verifies exact figure-data rows first.
"""
import math
from pathlib import Path
from experiments import priority35b_design as design

GROUPS = ('joint_c001', 'per_tensor_c001', 'joint_c01', 'joint_half', 'joint_med')
COLORS = ('#0072B2', '#666666', '#D55E00', '#009E73', '#CC79A7')
EPS_LABEL = {1:'1',3:'3',10:'10',30:'30',100:'100',1000:'1k',10000:'10k',1000000:'1M'}


def place_labels(fig, ax, labels):
    """Deterministic display-only collision avoidance with leader lines.

    Never move, omit or jitter measurements. Fail if all labels cannot fit.
    """
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=ax.get_window_extent(renderer)
    occupied=[]
    for label,x,y,color in labels:
        anchor=ax.transData.transform((x,y))
        candidates=[((i+.5)/12,(j+.5)/8) for i in range(12) for j in range(8)]
        candidates.sort(key=lambda p:sum((a-b)**2 for a,b in zip(ax.transAxes.transform(p),anchor)))
        placed=False
        for p in candidates:
            text=ax.text(*p,label,transform=ax.transAxes,fontsize=7,color=color,ha='center',va='center')
            box=text.get_window_extent(renderer).expanded(1.16,1.2)
            if bounds.contains(box.x0,box.y0) and bounds.contains(box.x1,box.y1) and not any(box.overlaps(b) for b in occupied):
                occupied.append(box)
                ax.annotate('',xy=(x,y),xytext=p,textcoords=ax.transAxes,
                            arrowprops=dict(arrowstyle='-',lw=.35,color=color,alpha=.55),zorder=1)
                placed=True;break
            text.remove()
        if not placed:raise ValueError('epsilon label layout does not fit; visual QA required')


def validate(rows, controls):
    seen=set()
    for r in rows:
        key=(r['dataset'],r['arm'])
        if key in seen or r['dataset'] not in design.DATASETS:raise ValueError('figure inventory')
        seen.add(key)
        if (r['dataset'],r['cell']) not in design.QUALIFIED:raise ValueError('unqualified plotted estimator')
        if r['utility_n']!=11 or r['leakage_n']!=39 or r['ranks']!=[13,27]:raise ValueError('figure units/intervals')
        vals=[r[k] for k in ('x','x_low','x_high','y','y_low','y_high')]
        if not all(math.isfinite(v) for v in vals):raise ValueError('non-finite figure')
        if not r['x_low']<=r['x']<=r['x_high'] or not 0<=r['y_low']<=r['y']<=r['y_high']<=1:
            raise ValueError('figure interval/range')
        if r['kind'] not in ('dp','clip_only','unprotected','v1','v2'):raise ValueError('figure arm kind')
        if r['kind'] in ('dp','clip_only') and r['group'] not in GROUPS:raise ValueError('figure clip group')
        if r['kind']=='dp' and r['epsilon'] not in design.EPSILONS:raise ValueError('epsilon label')
    if any(sum(r['dataset']==d for r in rows)!=32 for d in design.DATASETS):raise ValueError('all32 arms per panel')
    for d in design.DATASETS:
        cells={r['cell'] for r in rows if r['dataset']==d}
        if len(cells)!=1:raise ValueError('fixed estimator per panel')
        if d not in controls or set(controls[d])!={'mean_mode','empirical_marginal'}:raise ValueError('data-free lines')
        if not all(math.isfinite(v) and 0<=v<=1 for v in controls[d].values()):raise ValueError('data-free accuracy')


def make_figure(rows, controls, synthetic=False):
    validate(rows,controls)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    with plt.rc_context({'font.size':7,'axes.titlesize':7,'axes.labelsize':7,
                         'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42}):
        fig,axes=plt.subplots(3,1,sharex=True,sharey=True,figsize=(8.8/2.54,18/2.54))
        fig.subplots_adjust(left=.17,right=.97,bottom=.27,top=.94,hspace=.45)
        xmin=min(0,min(r['x_low'] for r in rows));xmax=max(r['x_high'] for r in rows)
        span=max(.05,xmax-xmin);xlims=(xmin-.03*span,xmax+.08*span)
        for ax,d in zip(axes,design.DATASETS):
            data=[r for r in rows if r['dataset']==d]
            labels=[]
            ax.set_title(d.replace('_','-').upper()+' | '+data[0]['cell'],loc='left')
            ax.set_xlim(xlims);ax.set_ylim(-3,108);ax.set_yticks((0,50,100))
            ax.grid(alpha=.18);ax.axvline(0,color='#999999',lw=.5)
            ax.axhline(100*controls[d]['mean_mode'],color='#333333',ls='--',lw=.8)
            ax.axhline(100*controls[d]['empirical_marginal'],color='#999999',ls=':',lw=.8)
            for group,color in zip(GROUPS,COLORS):
                points=sorted((r for r in data if r['kind']=='dp' and r['group']==group),key=lambda r:r['epsilon'])
                ax.plot([r['x'] for r in points],[100*r['y'] for r in points],color=color,lw=.7,alpha=.8)
            for r in data:
                kind=r['kind'];g=GROUPS.index(r['group']) if kind in ('dp','clip_only') else 0
                color=COLORS[g] if kind in ('dp','clip_only') else {'unprotected':'#111111','v1':'#56B4E9','v2':'#E69F00'}[kind]
                marker={'dp':'o','clip_only':'s','unprotected':'x','v1':'D','v2':'^'}[kind]
                ax.errorbar(r['x'],100*r['y'],xerr=[[r['x']-r['x_low']],[r['x_high']-r['x']]],
                            yerr=[[100*(r['y']-r['y_low'])],[100*(r['y_high']-r['y'])]],
                            fmt=marker,color=color,ms=3,lw=.45,capsize=1,alpha=.85)
                label=EPS_LABEL[r['epsilon']] if kind=='dp' else {'clip_only':'clip','unprotected':'none','v1':'v1','v2':'v2'}[kind]
                labels.append((label,r['x'],100*r['y'],color))
            place_labels(fig,ax,labels)
        axes[1].set_ylabel('Median feature accuracy (%)')
        axes[-1].set_xlabel('Utility loss = -mean paired Delta-F1')
        handles=[Line2D([0],[0],color=c,marker='o',lw=.7,label=l) for c,l in zip(COLORS,
                 ('Joint C=.01','Per-tensor C=.01','Joint C=.1','Joint C=median/2','Joint C=median'))]
        handles += [Line2D([0],[0],color='#333333',ls='--',label='Mean/mode'),
                    Line2D([0],[0],color='#999999',ls=':',label='Empirical marginal'),
                    Line2D([0],[0],color='#111111',marker='s',lw=0,label='Clip-only: no DP')]
        fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.12,.11),ncol=2,frameon=False,fontsize=7)
        fig.text(.17,.08,'Epsilon labels: k=1,000; M=1,000,000.\n'
                 'x: Config-C training, n=11, 95% t intervals.\n'
                 'y: P34C single gradient, n=39, ranks 13/27.\n'
                 'Axes are from different configurations.',fontsize=7,va='top')
        if synthetic:fig.suptitle('SYNTHETIC LAYOUT CHECK - NOT RESULTS',fontsize=7)
        fig.canvas.draw()
    return fig


def save_new(fig, directory):
    """Only new files; final publication caller must separately perform visual QA."""
    directory=Path(directory);pdf=directory/'frontier.pdf';png=directory/'frontier.png'
    if pdf.exists() or png.exists():raise ValueError('figure output exists; preserve original')
    directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(pdf,dpi=300)
    fig.savefig(png,dpi=300)
    return pdf,png
