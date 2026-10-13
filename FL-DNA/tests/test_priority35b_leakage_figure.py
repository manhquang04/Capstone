"""Synthetic layout fixtures only; no protected measurements loaded."""
import unittest
from experiments import priority35b_leakage_figure as f
from experiments import priority35b_design as d


def fixture():
    rows=[]
    for dataset in d.DATASETS:
        for group,eps in [('joint_c001',d.EPSILONS),('per_tensor_c001',d.EPSILONS),
                          ('joint_c01',(10,1000,1000000)),('joint_half',(10,1000,1000000)),
                          ('joint_med',(10,1000,1000000))]:
            for i,e in enumerate(eps):
                x=.02+.025*i+.008*f.GROUPS.index(group);y=.15+.08*i
                rows.append(dict(dataset=dataset,cell='ratio_batch1',arm=group+str(e),kind='dp',
                    group=group,epsilon=e,x=x,x_low=x-.005,x_high=x+.005,y=y,y_low=y-.03,y_high=y+.03,
                    utility_n=11,leakage_n=39,ranks=[13,27]))
        for i,group in enumerate(('joint_c001','joint_c01','joint_half','joint_med')):
            rows.append(dict(rows[-1],arm=group+'clip',kind='clip_only',group=group,epsilon=None,
                             x=.01+i*.02,x_low=.005+i*.02,x_high=.015+i*.02,y=.84,y_low=.8,y_high=.9))
        for i,kind in enumerate(('unprotected','v1','v2')):
            rows.append(dict(rows[-1],arm=kind,kind=kind,group=None,x=i*.005,x_low=i*.005-.003,
                             x_high=i*.005+.003,y=1-i*.2,y_low=1-i*.2-.03,y_high=1-i*.2))
    return rows,{dataset:dict(mean_mode=.3,empirical_marginal=.2) for dataset in d.DATASETS}


class Tests(unittest.TestCase):
    def test_inventory_and_finite(self):
        rows,controls=fixture();f.validate(rows,controls)
        with self.assertRaises(ValueError):f.validate(rows[:-1],controls)
        rows[0]['x']=float('nan')
        with self.assertRaises(ValueError):f.validate(rows,controls)

    def test_size_and_shared_axes(self):
        rows,controls=fixture();fig=f.make_figure(rows,controls,synthetic=True)
        self.assertAlmostEqual(fig.get_figwidth()*2.54,8.8)
        self.assertEqual(len(fig.axes),3)
        self.assertEqual(fig.axes[0].get_xlim(),fig.axes[2].get_xlim())
        renderer=fig.canvas.get_renderer()
        for ax in fig.axes:
            boxes=[text.get_window_extent(renderer) for text in ax.texts if text.get_text()]
            self.assertEqual(len(boxes),32)
            self.assertFalse(any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]))
        for text in fig.texts:
            box=text.get_window_extent(renderer)
            self.assertGreaterEqual(box.y0,0)
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__=='__main__':unittest.main()
