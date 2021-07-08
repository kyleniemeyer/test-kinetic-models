"""

.. moduleauthor:: Kyle Niemeyer <kyle.niemeyer@gmail.com>
"""

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
from .detect_peaks import detect_peaks

def first_derivative(x, y):
    """Evaluates first derivative using second-order finite differences.

    Uses (second-order) centeral difference in interior and second-order
    one-sided difference at boundaries.

    :param x: Independent variable array
    :type x: numpy.ndarray
    :param y: Dependent variable array
    :type y: numpy.ndarray
    :return: First derivative, :math:`dy/dx`
    :rtype: numpy.ndarray
    """
    return numpy.gradient(y, x, edge_order=2)


def sample_rising_pressure(time_end, init_pres, freq, pressure_rise_rate):
    """Samples pressure for particular frequency assuming linear rise.

    :param float time_end: End time of simulation in s
    :param float init_pres: Initial pressure
    :param float freq: Frequency of sampling, in Hz
    :param float pressure_rise_rate: Pressure rise rate, in s^-1
    :return: List of times and pressures
    :rtype: list of numpy.ndarray
    """
    times = numpy.arange(0.0, time_end + (1.0 / freq), (1.0 / freq))
    pressures = init_pres * (pressure_rise_rate * times + 1.0)
    return [times, pressures]


def create_volume_history(mech, temp, pres, reactants, pres_rise, time_end):
    """Constructs a volume profile based on intiial conditions and pressure rise.

    :param str mech: Cantera-format mechanism file
    :param float temp: Initial temperature in K
    :param float pres: Initial pressure in Pa
    :param str reactants: Reactants composition in mole fraction
    :param float pres_rise: Pressure rise rate, in s^-1
    :param float time_end: End time of simulation in s
    :return: List of times and volumes
    :rtype: list of numpy.ndarray
    """
    gas = ct.Solution(mech)
    gas.TPX = temp, pres, reactants
    initial_entropy = gas.entropy_mass
    initial_density = gas.density

    # Sample pressure at 20 kHz
    freq = 2.0e4
    [times, pressures] = sample_rising_pressure(time_end, pres, freq, pres_rise)

    # Calculate volume profile based on pressure
    volumes = numpy.zeros((len(pressures)))
    for i, p in enumerate(pressures):
        gas.SP = initial_entropy, p
        volumes[i] = initial_density / gas.density

    return [times, volumes]


class VolumeProfile(object):
    """Set the velocity of reactor moving wall via specified volume profile.

    The initialization and calling of this class are handled by the
    `Func1
    <http://cantera.github.io/docs/sphinx/html/cython/zerodim.html#cantera.Func1>`_
    interface of Cantera.

    Based on ``VolumeProfile`` implemented in Bryan W. Weber's
    `CanSen <http://bryanwweber.github.io/CanSen/>`
    """

    def __init__(self, volume_history):
        """Set the initial values of the arrays from the input keywords.

        The time and volume are read from the input file and stored in an
        ``VolumeHistory`` object. The velocity is calculated by
        assuming a unit area and using central differences. This function is
        only called once when the class is initialized at the beginning of a
        problem so it is efficient.

        :param VolumeHistory volume_history: time and volume history
        """

        # The time and volume are each stored as a ``numpy.array`` in the
        # properties dictionary. The volume is normalized by the first volume
        # element so that a unit area can be used to calculate the velocity.
        self.times = volume_history.time.magnitude
        volumes = (volume_history.quantity.magnitude /
                   volume_history.quantity.magnitude[0]
                   )

        # The velocity is calculated by the second-order central differences.
        self.velocity = first_derivative(self.times, volumes)

    def __call__(self, time):
        """Return (interpolated) velocity when called during a time step.

        :param float time: Current simulation time in seconds
        :return: Velocity in meters per second
        :rtype: float
        """
        return numpy.interp(time, self.times, self.velocity, left=0., right=0.)


class PressureRiseProfile(VolumeProfile):
    r"""Set the velocity of reactor moving wall via specified pressure rise.

    The initialization and calling of this class are handled by the
    `Func1 <http://cantera.github.io/docs/sphinx/html/cython/zerodim.html#cantera.Func1>`_
    interface of Cantera.

    The approach used here is based on that discussed by Chaos and Dryer,
    "Chemical-kinetic modeling of ignition delay: Considerations in
    interpreting shock tube data", *Int J Chem Kinet* 2010 42:143-150,
    `doi:10.1002/kin.20471 <http://dx.doi.org/10.1002/kin.20471`.
    A time-dependent polytropic state change is emulated by determining volume
    as a function of time, via a constant linear pressure rise :math:`A`
    (given as a percentage of the initial pressure):

    .. math::
       \frac{dv}{dt} &= -\frac{1}{\gamma} \frac{v(t)}{P(t)} \frac{dP}{dt} \\
       v(t) &= \frac{1}{\rho} \left[ \frac{P(t)}{P_0} \right]^{-1 / \gamma}

       \frac{dP}{dt} &= A P_0 \\
       \therefore P(t) &= P_0 (A t + 1)

       \frac{dv}{dt} = -A \frac{1}{\rho \gamma} (A t + 1)^{-1 / \gamma}

    The expression for :math:`\frac{dv}{dt}` can then be used directly for
    the ``Wall`` velocity.
    """

    def __init__(self, mech_filename, initial_temp, initial_pres,
                 reactants, pressure_rise, time_end
                 ):
        """Set the initial values of properties needed for velocity.

        :param str mech_filename: Cantera-format mechanism
        :param float initial_temp: Initial temperature in K
        :param float initial_pres: Initial pressure in Pa
        :param str reactants: Reactants composition in mole fraction
        :param float pres_rise: Pressure rise rate in s^-1
        :param float time_end: End time of simulation in s
        """

        [self.times, volumes] = create_volume_history(
                    mech_filename, initial_temp, initial_pres,
                    reactants, pressure_rise, time_end
                    )

        # Calculate velocity by second-order finite difference
        self.velocity = first_derivative(self.times, volumes)


class AutoIgnitionSimulation(object):
    """Class for ignition delay simulations."""

    def __init__(self, kind, apparatus, meta, properties):
        """Initialize simulation case.

        :param kind: Kind of experiment (e.g., 'ignition delay')
        :type kind: str
        :param apparatus: Type of apparatus ('shock tube' or 'rapid compression machine')
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

        self.gas = ct.Solution(model_file)

        # Convert ignition delay to seconds
        self.properties.ignition_delay.ito('second')

        # Set end time of simulation to 100 times the experimental ignition delay
        if hasattr(self.properties.ignition_delay, 'value'):
            self.time_end = 100. * self.properties.ignition_delay.value.magnitude
        else:
            self.time_end = 100. * self.properties.ignition_delay.magnitude

        # Initial temperature needed in Kelvin for Cantera
        self.properties.temperature.ito('kelvin')

        # Initial pressure needed in Pa for Cantera
        self.properties.pressure.ito('pascal')

        # convert reactant names to those needed for model
        reactants = [species_key[self.properties.composition[spec].species_name] + ':' +
                     str(self.properties.composition[spec].amount.magnitude)
                     for spec in self.properties.composition
                     ]
        reactants = ','.join(reactants)

        # need to extract values from Quantity or Measurement object
        if hasattr(self.properties.temperature, 'value'):
            temp = self.properties.temperature.value.magnitude
        elif hasattr(self.properties.temperature, 'nominal_value'):
            temp = self.properties.temperature.nominal_value
        else:
            temp = self.properties.temperature.magnitude
        if hasattr(self.properties.pressure, 'value'):
            pres = self.properties.pressure.value.magnitude
        elif hasattr(self.properties.pressure, 'nominal_value'):
            temp = self.properties.pressure.nominal_value
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

        # Create non-interacting ``Reservoir`` on other side of ``Wall``
        env = ct.Reservoir(ct.Solution('air.xml'))

        # All reactors are ``IdealGasReactor`` objects
        self.reac = ct.IdealGasReactor(self.gas)
        if self.apparatus == 'shock tube' and self.properties.pressure_rise is None:
            # Shock tube modeled by constant UV
            self.wall = ct.Wall(self.reac, env, A=1.0, velocity=0)

        elif self.apparatus == 'shock tube' and self.properties.pressure_rise is not None:
            # Shock tube modeled by constant UV with isentropic compression

            # Need to convert pressure rise units to seconds
            self.properties.pressure_rise.ito('1 / second')
            if hasattr(self.properties.pressure_rise, 'value'):
                pres_rise = self.properties.pressure_rise.value.magnitude
            else:
                pres_rise = self.properties.pressure_rise.magnitude

            self.wall = ct.Wall(self.reac, env, A=1.0,
                                velocity=PressureRiseProfile(
                                    model_file,
                                    self.gas.T,
                                    self.gas.P,
                                    self.gas.X,
                                    pres_rise,
                                    self.time_end
                                    )
                                )

        elif (self.apparatus == 'rapid compression machine' and
              self.properties.volume_history is None
              ):
            # Rapid compression machine modeled by constant UV
            self.wall = ct.Wall(self.reac, env, A=1.0, velocity=0)

        elif (self.apparatus == 'rapid compression machine' and
              self.properties.volume_history is not None
              ):
            # Rapid compression machine modeled with volume-time history

            # First convert time units if necessary
            self.properties.volume_history.time.ito('second')

            self.wall = ct.Wall(self.reac, env, A=1.0,
                                velocity=VolumeProfile(self.properties.volume_history)
                                )

        # Number of solution variables is number of species + mass,
        # volume, temperature
        self.n_vars = self.reac.kinetics.n_species + 3

        # Create ``ReactorNet`` newtork
        self.reac_net = ct.ReactorNet([self.reac])

        # Set maximum time step based on volume-time history, if present
        if self.properties.volume_history is not None:
            # Minimum difference between volume profile times
            min_time = numpy.min(numpy.diff(self.properties.volume_history.time.magnitude))
            self.reac_net.set_max_time_step(min_time)

        # Check if species ignition target, that species is present.
        if self.properties.ignition_type['target'] not in ['pressure', 'temperature']:
            # Other targets are species
            spec = self.properties.ignition_type['target']

            # Try finding species in upper- and lower-case
            try_list = [spec, spec.lower(), spec.upper()]

            # If excited radical, may need to fall back to nonexcited species
            if spec[-1] == '*':
                try_list += [spec[:-1], spec[:-1].lower(), spec[:-1].upper()]

            ind = None
            for sp in try_list:
                try:
                    ind = self.gas.species_index(sp)
                    break
                except ValueError:
                    pass

            # store index of target species
            if ind:
                self.properties.ignition_target = ind
                self.properties.ignition_type = self.properties.ignition_type['type']
            else:
                warnings.warn(
                    spec + ' not found in model; falling back on pressure.',
                    RuntimeWarning
                    )
                self.properties.ignition_target = 'pressure'
                self.properties.ignition_type = 'd/dt max'
        else:
            self.properties.ignition_target = self.properties.ignition_type['target']
            self.properties.ignition_type = self.properties.ignition_type['type']

        # Set file for later data file
        file_path = os.path.join(path, self.meta['id'] + '.h5')
        self.meta['save-file'] = file_path

    def run_case(self, restart=False):
        """Run simulation case set up ``setup_case``.

        :param bool restart: If ``True``, skip if results file exists.
        """

        if restart and os.path.isfile(self.meta['save-file']):
            print('Skipped existing case ', self.meta['id'])
            return

        # Save simulation results in hdf5 table format.
        table_def = {'time': tables.Float64Col(pos=0),
                     'temperature': tables.Float64Col(pos=1),
                     'pressure': tables.Float64Col(pos=2),
                     'volume': tables.Float64Col(pos=3),
                     'mass_fractions': tables.Float64Col(
                          shape=(self.reac.thermo.n_species), pos=4
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
            timestep['time'] = self.reac_net.time
            timestep['temperature'] = self.reac.T
            timestep['pressure'] = self.reac.thermo.P
            timestep['volume'] = self.reac.volume
            timestep['mass_fractions'] = self.reac.Y
            # Add ``timestep`` to table
            timestep.append()

            # Main time integration loop; continue integration while time of
            # the ``ReactorNet`` is less than specified end time.
            while self.reac_net.time < self.time_end:
                self.reac_net.step()

                # Save new timestep information
                timestep['time'] = self.reac_net.time
                timestep['temperature'] = self.reac.T
                timestep['pressure'] = self.reac.thermo.P
                timestep['volume'] = self.reac.volume
                timestep['mass_fractions'] = self.reac.Y

                # Add ``timestep`` to table
                timestep.append()

            # Write ``table`` to disk
            table.flush()

        print('Done with case ', self.meta['id'])

    def process_results(self):
        """Process integration results to obtain ignition delay.
        """

        # Load saved integration results
        with tables.open_file(self.meta['save-file'], 'r') as h5file:
            # Load Table with Group name simulation
            table = h5file.root.simulation

            time = table.col('time')
            if self.properties.ignition_target == 'pressure':
                target = table.col('pressure')
            elif self.properties.ignition_target == 'temperature':
                target = table.col('temperature')
            else:
                target = table.col('mass_fractions')[:, self.properties.ignition_target]

        # add units to time
        time = time * units.second

        # Analysis for ignition depends on type specified
        if self.properties.ignition_type in ['max', 'd/dt max']:
            if self.properties.ignition_type == 'd/dt max':
                # Evaluate derivative
                target = first_derivative(time.magnitude, target)

            # Get indices of peaks
            ind = detect_peaks(target)

            # Fall back on derivative if max value doesn't work.
            if len(ind) == 0 and self.properties.ignition_type == 'max':
                target = first_derivative(time.magnitude, target)
                ind = detect_peaks(target)

            # something has gone wrong if there is still no peak
            if len(ind) == 0:
                filename = 'target-data-' + self.meta['id'] + '.out'
                warnings.warn('No peak found, dumping target data to ' +
                              filename + ' and continuing',
                              RuntimeWarning
                              )
                numpy.savetxt(filename, numpy.c_[time.magnitude, target],
                              header=('time, target ('+self.properties.ignition_target+')')
                              )
                self.meta['simulated-ignition-delay'] = 0.0 * units.second
                return


            # Get index of largest peak (overall ignition delay)
            max_ind = ind[numpy.argmax(target[ind])]

            # Will need to subtract compression time for RCM
            time_comp = 0.0
            if hasattr(self.properties.rcm_data, 'compression_time'):
                if hasattr(self.properties.rcm_data.compression_time, 'value'):
                    time_comp = self.properties.rcm_data.compression_time.value
                else:
                    time_comp = self.properties.rcm_data.compression_time


            ign_delays = time[ind[numpy.where((time[ind[ind <= max_ind]] - time_comp)
                                              > 0. * units.second
                                             )]] - time_comp
        elif self.properties.ignition_type == '1/2 max':
            # maximum value, and associated index
            max_val = numpy.max(target)
            ind = detect_peaks(target)
            max_ind = ind[numpy.argmax(target[ind])]

            # TODO: interpolate for actual half-max value
            # Find index associated with the 1/2 max value, but only consider
            # points before the peak
            half_idx = (numpy.abs(target[0:max_ind] - 0.5 * max_val)).argmin()
            ign_delays = [time[half_idx]]

            # TODO: detect two-stage ignition when 1/2 max type?

        # Overall ignition delay
        if len(ign_delays) > 0:
            self.meta['simulated-ignition-delay'] = ign_delays[-1]
        else:
            self.meta['simulated-ignition-delay'] = 0.0 * units.second

        # First-stage ignition delay
        if len(ign_delays) > 1:
            self.meta['simulated-first-stage-delay'] = ign_delays[0]
        else:
            self.meta['simulated-first-stage-delay'] = numpy.nan * units.second
