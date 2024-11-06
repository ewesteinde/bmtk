import numpy as np
from neuron import h

from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.simulator.bionet.io_tools import io
from bmtk.simulator.core.modules import iclamp


class SEClamp(SimulatorMod):
    def __init__(self, input_type, **mod_args):
        
        # Select location to place iclamp, if not specified use the center of the soma
        self._section_name = mod_args.get('section_name', 'soma')
        self._section_index = mod_args.get('section_index', 0)
        self._section_dist = mod_args.get('section_dist', 0.5)

        self._rs = mod_args.get('resistance', 1.0)
        self._vc = mod_args.get('vc', 0.0)

        self._delay = mod_args['delay']
        self._amp = mod_args['amp']
        self._duration = mod_args['duration']
        self._ton = self._delay
        self._toff = self._delay + self._duration

        # Check f section_index is a range (ie. "section_index": [500.0, 1000.0])
        self._ranged_index = isinstance(self._section_index, (list, tuple))

        # SEClamp objects need to be saved in memory otherwise NRN will try to garbage collect them
        # prematurly
        self._seclamp = None

        self._node_set = mod_args.get('node_set', 'all') 
        
    def initialize(self, sim):
        select_gids = list(sim.net.get_node_set(self._node_set).gids())
        gids_on_rank = list(set(select_gids) & set(select_gids))

        for gid in gids_on_rank:           
            cell = sim.net.get_cell_gid(gid)

            if self._ranged_index:
                hobj_sec = self._find_section(cell)
            else:
                hobj_sec = getattr(cell.hobj, self._section_name)[self._section_index](self._section_dist)
            
            self._seclamp = self.create_clamp(hobj_sec)

    def step(self, sim, tstep):
        if self._ton <= sim.dt*tstep <= self._toff:
            self._seclamp.rs = self._rs
            # print(sim.dt*tstep)
        else:
            self._seclamp.rs = 10.0e20
            

    def create_clamp(self, hobj):
        stim = h.SEClamp(hobj)

        stim.dur1 = self._delay
        stim.dur2 = self._duration
        stim.dur3 = 0.0

        stim.amp1 = 0.0
        stim.amp2 = self._amp
        stim.amp3 = 0.0
        stim.vc = self._vc

        return stim
        