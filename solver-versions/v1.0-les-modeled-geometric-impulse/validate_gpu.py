"""Run after building the DLL/.so. This checks REAL CUDA, unlike the host checks."""
from pathlib import Path
import argparse
import importlib.util
import numpy as np
from main import PARAMETERS
from python_api import initialize, advance, diagnostics, copy_field, close


def run(reference_path=None):
    parameters=dict(PARAMETERS)
    parameters.update(nx=32,ny=24,nz=4,lx=8.,ly=6.,body_x=2.,probe_x=2.,stats_start=0.)
    reference_module=None
    if reference_path:
        spec=importlib.util.spec_from_file_location("reference",reference_path)
        reference_module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference_module)
    for mode,les,particles,adaptive,high_order in [(0,1,0,0,0),(0,0,0,1,1),(1,1,0,0,0),(2,0,0,0,0),(0,0,1,0,0)]:
        parameters.update(mode=mode,les=les,particle_advection=particles,adaptive=adaptive,adapt_cfl=adaptive,high_order=high_order)
        reference=None
        if reference_module:
            reference_module.__dict__.update(BACKEND="cpu",CELLS_PER_D=4,SPAN_CELLS=4,LX_D=8.,LY_D=6.,LZ_D=np.pi,X_BODY_D=2.,PROBE_X_D=2.,N_STEPS=10,STATS_START_T=0.,MODE=["impulse","eulerian","eulerian_vc"][mode],LES_ON=bool(les),IMPULSE_ADVECTION="pic_flip" if particles else "semi_lagrangian",ADAPTIVE_DT=bool(adaptive),ADAPT_CFL=bool(adaptive),HIGH_ORDER_STRETCH_GRAD=bool(high_order))
            reference=reference_module.BluffBody3D()
        handle,config=initialize(parameters)
        try:
            for step in range(3):
                result=advance(handle,1)
                assert np.isfinite(result["energy"])
                if reference:
                    reference.step()
                    for name in ("u","v","w","mx","my","mz","phi"):
                        np.testing.assert_allclose(copy_field(handle,config,name),getattr(reference,name),atol=3e-5,rtol=3e-5)
            result=diagnostics(handle)
            assert result["divergence_rms"]<3e-5, result
            assert result["wall_slip"]<3e-6, result
            assert result["min_determinant"]>0, result
            for name in ("u","v","w","mx","my","mz","phi","nut"):
                assert np.isfinite(copy_field(handle,config,name)).all()
            print(f"PASS mode={mode}, LES={les}, particles={particles}, adaptive={adaptive}, high_order={high_order}")
        finally:
            close(handle)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--reference",type=Path,help="Optional original 3-D Python solver (with BluffBody3D)")
    run(parser.parse_args().reference)
