import os, sys

from bmtk.simulator import pointnet
import nest

# nest.ResetKernel()
# nest.Install('nestml_39f5da2aef404887ac26009e1449206e_module')


# neuron = nest.Create('ornstein_uhlenbeck_noise_neuron')
# print(list(neuron))

def run(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    network = pointnet.PointNetwork.from_config(configure)
    # network.add_nest_module('/local1/workspace/bmtk/examples/point_nestml_izh/components/nestml/')
    # network.add_nest_module('nestml_izh_module')
    # network.add_nest_module('/local1/workspace/bmtk/examples/point_nestml_izh/components/nestml/nestml_izh_module.so')
    
    
    # nest.Install('nestml_39f5da2aef404887ac26009e1449206e_module')
    sim = pointnet.PointSimulator.from_config(configure, network)
    sim.run()


if __name__ == '__main__':
    # Find the appropriate config.json file
    run('config.simulation.json')
    # run('config.simulation_perturbations.json')
