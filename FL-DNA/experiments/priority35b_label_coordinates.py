"""Presentation-only cap-coordinate normalization; frozen predecessor unchanged."""
from pathlib import Path
import numpy as np
from experiments import priority35b_presentation_finalize as presentation

FREEZE='artifacts/priority35b/b3/label_coordinate_freeze.json'


def verify():
    d=presentation.i.read(Path(presentation.ROOT)/FREEZE)
    for f,h in {**d['sources'],**d['inputs']}.items():
        if presentation.i.sha(Path(presentation.ROOT)/f)!=h:raise ValueError('label adapter drift')
    return d


def labels(fig,ax,values):
    from matplotlib.transforms import Bbox
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=ax.get_window_extent(renderer);occupied=[]
    for line in ax.lines:
        if line.get_marker() not in ('None',None,''):
            # Display-only normalization; no artist data is assigned or modified.
            coordinates=np.asarray(np.column_stack((line.get_xdata(),line.get_ydata())),dtype=np.float64)
            for x,y in ax.transData.transform(coordinates):occupied.append(Bbox.from_extents(x-5,y-5,x+5,y+5))
    for col in ax.collections:
        for seg in col.get_segments():
            coordinates=ax.transData.transform(np.asarray(seg,dtype=np.float64))
            occupied.append(Bbox.from_extents(coordinates[:,0].min()-2,coordinates[:,1].min()-2,
                                             coordinates[:,0].max()+2,coordinates[:,1].max()+2))
    for label,x,y,color in values:
        anchor=ax.transData.transform((x,y))
        candidates=[((u+.5)/28,(v+.5)/16) for u in range(28) for v in range(16)]
        candidates.sort(key=lambda z:sum((a-b)**2 for a,b in zip(ax.transAxes.transform(z),anchor)))
        for z in candidates:
            text=ax.text(*z,label,transform=ax.transAxes,fontsize=7,color=color,ha='center',va='center')
            box=text.get_window_extent(renderer).expanded(1.22,1.35)
            if bounds.contains(box.x0,box.y0) and bounds.contains(box.x1,box.y1) and not any(box.overlaps(b) for b in occupied):
                occupied.append(box)
                ax.annotate('',xy=(x,y),xytext=z,textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle='-',lw=.35,color=color,alpha=.55),zorder=1)
                break
            text.remove()
        else:raise ValueError('label layout does not fit')


def main():
    verify();old=presentation.install
    def install():
        old();base=presentation.audit.verify_freeze
        def augmented():
            d=base();f=verify();d['files'].update(f['sources']);d['files'].update(f['inputs'])
            d['files'][FREEZE]=presentation.i.sha(Path(presentation.ROOT)/FREEZE)
            return d
        presentation.audit.verify_freeze=augmented
    presentation.install=install;presentation.label_positions=labels;presentation.main()


if __name__=='__main__':main()
