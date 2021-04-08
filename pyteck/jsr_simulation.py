#Documentation header!!!

# Python 2 compatibility
from __future__ import print_function
from __future__ import division

# Standard libraries
import os
from collections import namedtuple
import warnings
import numpy

# Related modules
try:
    import cantera as ct
    ct.suppress_thermo_warnings()
except ImportError:
    print("Error: Cantera must be installed.")
    raise
try:
    import tables
except ImportError:
    print('PyTables must be installed')
    raise

# Local imports
from .utils import units

class JSRSimulation(object):
    """Class for jet-stirred reactor simulations."""

    def __init__(self, kind, apparatus, meta, properties):
        """Initialize simulation case.

        :param kind: Kind of experiment (e.g., 'species profile')
        :type kind: str
        :param apparatus: Type of apparatus ('jet-stirred reactor')
        :type apparatus: str
        :param meta: some metadata for this case
        :type meta: dict
        :param properties: set of properties for this case
        :type properties: pyked.chemked.DataPoint
        """
        self.kind = kind
        self.apparatus = apparatus
        self.meta = meta
        self.properties = properties

    def setup_case(self, model_file, species_key, path=''):
        """Sets up the simulation case to be run.

        :param str model_file: Filename for Cantera-format model
        :param dict species_key: Dictionary with species names for `model_file`
        :param str path: Path for data file
        """
        # Establishes the model
        self.gas = ct.Solution(model_file)

        # Set max simulation time, pressure valve coefficient, and max pressure rise for Cantera
        # These could be set to something in ChemKED file, but haven't seen these specified at all....
        self.maxsimulationtime = 50
        self.pressurevalcof = 0.01
        self.maxpressurerise = 0.01
        
        # Reactor volume needed in m^3 for Cantera
        self.apparatus.volume.ito('m^3')
        
        # Residence time needed in s for Cantera
        self.apparatus.restime.ito('s')

        # Initial temperature needed in Kelvin for Cantera
        self.properties.temperature.ito('kelvin')

        # Initial pressure needed in Pa for Cantera
        self.properties.pressure.ito('pascal')

        # Convert reactant names to those needed for model
        reactants = [species_key[self.properties.composition[spec].species_name] + ':' +
                     str(self.properties.composition[spec].amount.magnitude)
                     for spec in self.properties.composition
                     ]
        reactants = ','.join(reactants)

        # Need to extract values from quantity or measurement object
        if hasattr(self.properties.temperature, 'value'):
            temp = self.properties.temperature.value.magnitude
        elif hasattr(self.properties.temperature, 'nominal_value'):
            temp = self.properties.temperature.nominal_value
        else:
            temp = self.properties.temperature.magnitude
        if hasattr(self.properties.pressure, 'value'):
            pres = self.properties.pressure.value.magnitude
        elif hasattr(self.properties.pressure, 'nominal_value'):
            pres = self.properties.pressure.nominal_value
        else:
            pres = self.properties.pressure.magnitude

        # Reactants given in format for Cantera
        if self.properties.composition_type in ['mole fraction', 'mole percent']:
            self.gas.TPX = temp, pres, reactants
        elif self.properties.composition_type == 'mass fraction':
            self.gas.TPY = temp, pres, reactants
        else:
            raise(BaseException('error: not supported'))
            return
        
        # Upstream and exhaust
        self.fuelairmix = ct.Reservoir(self.gas)
        self.exhaust = ct.Reservoir(self.gas)

        # Ideal gas reactor 
        self.reactor = ct.IdealGasReactor(self.gas, energy='off', volume=self.volume)
        self.massflowcontrol = ct.MassFlowController(upstream=self.fuelairmix,downstream=self.reactor,mdot=self.reactor.mass/self.restime)
        self.pressureregulator = ct.Valve(upstream=self.reactor,downstream=self.exhaust,K=self.pressurevalcof)

        # Create reactor newtork
        self.reactor_net = ct.ReactorNet([self.reactor])

        # Set file path
        file_path = os.path.join(path, self.meta['id'] + '.h5')
        self.meta['save-file'] = file_path

    def run_case(self, restart=False):
        """Run simulation case set up ``setup_case``.

        :param bool restart: If ``True``, skip if results file exists.
        """

        if restart and os.path.isfile(self.meta['save-file']):
            print('Skipped existing case ', self.meta['id'])
            return

        # Save simulation results in hdf5 table format
        table_def = {'time': tables.Float64Col(pos=0),
                     'temperature': tables.Float64Col(pos=1),
                     'pressure': tables.Float64Col(pos=2),
                     'volume': tables.Float64Col(pos=3),
                     'mole_fractions': tables.Float64Col(
                          shape=(self.reactor.thermo.n_species), pos=4
                          ),
                     }

        with tables.open_file(self.meta['save-file'], mode='w',
                              title=self.meta['id']
                              ) as h5file:

            table = h5file.create_table(where=h5file.root,
                                        name='simulation',
                                        description=table_def
                                        )
            # Row instance to save timestep information to
            timestep = table.row
            # Save initial conditions
            timestep['time'] = self.reactor_net.time
            timestep['temperature'] = self.reactor.T
            timestep['pressure'] = self.reactor.thermo.P
            timestep['volume'] = self.reactor.volume
            timestep['mole_fractions'] = self.reactor.X
            # Add ``timestep`` to table
            timestep.append()

            # Main time integration loop; continue integration while time of
            # the ``ReactorNet`` is less than specified end time.
            while self.reac_net.time < self.maxsimulationtime:
                self.reactor_net.step()

                # Save new timestep information
                timestep['time'] = self.reactor_net.time
                timestep['temperature'] = self.reactor.T
                timestep['pressure'] = self.reactor.thermo.P
                timestep['volume'] = self.reactor.volume
                timestep['mole_fractions'] = self.reactor.X

                # Add ``timestep`` to table
                timestep.append()

            # Write ``table`` to disk
            table.flush()

        print('Done with case ', self.meta['id'])

    def process_results(self):
        """Process integration results to obtain concentrations.
        """
        
        ## concentrations need to be compiled, maybe with species names??

        # Load saved integration results
        with tables.open_file(self.meta['save-file'], 'r') as h5file:
            # Load Table with Group name simulation
            table = h5file.root.simulation
            
        self.meta['simulated species profile'] = table.col('mole_fractions')