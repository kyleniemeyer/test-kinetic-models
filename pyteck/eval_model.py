# Python 2 compatibility
from __future__ import print_function
from __future__ import division

# Standard libraries
import os
from os.path import splitext, basename
import multiprocessing
import warnings

import numpy
from scipy.interpolate import UnivariateSpline

try:
    import yaml
except ImportError:
    print('Warning: YAML must be installed to read input file.')
    raise

from pyked.chemked import ChemKED, IgnitionDataPoint  

# Local imports
from .utils import units
from .autoignition_simulation import AutoIgnitionSimulation as Simulation

min_deviation = 0.10
"""float: minimum allowable standard deviation for experimental data"""


def create_simulations(dataset, properties):
    """Set up individual simulations for each ignition delay value.

    Parameters
    ----------
    dataset :

    properties : pyked.chemked.ChemKED
        ChemKED object with full set of experimental properties

    Returns
    -------
    simulations : list
        List of :class:`AutoignitionSimulation` objects for each simulation

    """

    simulations = []
    for idx, case in enumerate(properties.datapoints):
        sim_meta = {}
        # Common metadata
        sim_meta['data-file'] = dataset
        sim_meta['id'] = splitext(basename(dataset))[0] + '_' + str(idx)

        simulations.append(Simulation(properties.experiment_type,
                                      properties.apparatus.kind,
                                      sim_meta,
                                      case
                                      )
                           )
    return simulations

def simulation_worker(sim_tuple):
    """Worker for multiprocessing of simulation cases.

    Parameters
    ----------
    sim_tuple : tuple
        Contains AutoignitionSimulation object and other parameters needed to setup
        and run case.

    Returns
    -------
    sim : ``AutoignitionSimulation``
        AutoignitionSimulation case with calculated ignition delay.

    """
    sim, model_file, model_spec_key, path, restart = sim_tuple

    sim.setup_case(model_file, model_spec_key, path)
    sim.run_case(restart)

    sim = Simulation(sim.kind, sim.apparatus, sim.meta, sim.properties)
    return sim


def estimate_std_dev(indep_variable, dep_variable):
    """

    Parameters
    ----------
    indep_variable : ndarray, list(float)
        Independent variable (e.g., temperature, pressure)
    dep_variable : ndarray, list(float)
        Dependent variable (e.g., ignition delay)

    Returns
    -------
    standard_dev : float
        Standard deviation of difference between data and best-fit line

    """

    assert len(indep_variable) == len(dep_variable), \
        'independent and dependent variables not the same length'

    # ensure no repetition of independent variable by taking average of associated dependent
    # variables and removing duplicates
    vals, count = numpy.unique(indep_variable, return_counts=True)
    repeated = vals[count > 1]
    for val in repeated:
        idx, = numpy.where(indep_variable == val)
        dep_variable[idx[0]] = numpy.mean(dep_variable[idx])
        dep_variable = numpy.delete(dep_variable, idx[1:])
        indep_variable = numpy.delete(indep_variable, idx[1:])


    # ensure data sorted based on independent variable to avoid some problems
    sorted_vars = sorted(zip(indep_variable, dep_variable))
    indep_variable = [pt[0] for pt in sorted_vars]
    dep_variable = [pt[1] for pt in sorted_vars]

    # spline fit of the data
    if len(indep_variable) == 1 or len(indep_variable) == 2:
        # Fit of data will be perfect
        return min_deviation
    elif len(indep_variable) == 3:
        spline = UnivariateSpline(indep_variable, dep_variable, k=2)
    else:
        spline = UnivariateSpline(indep_variable, dep_variable)

    standard_dev = numpy.std(dep_variable - spline(indep_variable))

    if standard_dev < min_deviation:
        print('Standard deviation of {:.2f} too low, '
              'using {:.2f}'.format(standard_dev, min_deviation))
        standard_dev = min_deviation

    return standard_dev


def get_changing_variable(cases):
    """Identify variable changing across multiple cases.

    Parameters
    ----------
    cases : list(pyked.chemked.IgnitionDataPoint)
        List of IgnitionDataPoint with experimental case data.

    Returns
    -------
    variable : list(float)
        List of floats representing changing experimental variable.

    """
    changing_var = None

    for var_name in ['temperature', 'pressure']:
        if var_name == 'temperature':
            variable = [case.temperature for case in cases]
        elif var_name == 'pressure':
            variable = [case.pressure for case in cases]

        if not all([x == variable[0] for x in variable]):
            if not changing_var:
                changing_var = var_name
            else:
                warnings.warn('Warning: multiple changing variables. '
                              'Using temperature.',
                              RuntimeWarning
                              )
                changing_var = 'temperature'
                break

    # Temperature is default
    if changing_var is None:
        changing_var = 'temperature'

    if changing_var == 'temperature':
        variable = [case.temperature.value.magnitude if hasattr(case.temperature, 'value')
                    else case.temperature.magnitude
                    for case in cases
                    ]
    elif changing_var == 'pressure':
        variable = [case.pressure.value.magnitude if hasattr(case.pressure, 'value')
                    else case.pressure.magnitude
                    for case in cases
                    ]
    return variable


def evaluate_model(model_name, spec_keys_file, dataset_file,
                   data_path='data', model_path='models',
                   results_path='results', model_variant_file=None,
                   num_threads=None, print_results=False, restart=False,
                   skip_validation=False,
                   ):
    """Evaluates the ignition delay error of a model for a given dataset.

    Parameters
    ----------
    model_name : str
        Chemical kinetic model filename
    spec_keys_file : str
        Name of YAML file identifying important species
    dataset_file : str
        Name of file with list of data files
    data_path : str
        Local path for data files. Optional; default = 'data'
    model_path : str
        Local path for model file. Optional; default = 'models'
    results_path : str
        Local path for creating results files. Optional; default = 'results'
    model_variant_file : str
        Name of YAML file identifying ranges of conditions for variants of the
        kinetic model. Optional; default = ``None``
    num_threads : int
        Number of CPU threads to use for performing simulations in parallel.
        Optional; default = ``None``, in which case the available number of
        cores minus one is used.
    print_results : bool
        If ``True``, print results of the model evaluation to screen.
    restart : bool
        If ``True``, process saved results. Mainly intended for testing/development.
    skip_validation : bool
        If ``True``, skips validation of ChemKED files.

    Returns
    -------
    output : dict
        Dictionary with all information about model evaluation results.

    """
    # Create results_path if it doesn't exist
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Dict to translate species names into those used by models
    with open(spec_keys_file, 'r') as f:
        model_spec_key = yaml.safe_load(f)

    # Keys for models with variants depending on pressure or bath gas
    model_variant = None
    if model_variant_file:
        with open(model_variant_file, 'r') as f:
            model_variant = yaml.safe_load(f)

    # Read dataset list
    with open(dataset_file, 'r') as f:
        dataset_list = f.read().splitlines()

    error_func_sets = numpy.zeros(len(dataset_list))
    dev_func_sets = numpy.zeros(len(dataset_list))

    # Dictionary with all output data
    output = {'model': model_name, 'datasets': []}

    # If number of threads not specified, use either max number of available
    # cores minus 1, or use 1 if multiple cores not available.
    if not num_threads:
        num_threads = multiprocessing.cpu_count()-1 or 1

    # Loop through all datasets
    for idx_set, dataset in enumerate(dataset_list):

        dataset_meta = {'dataset': dataset, 'dataset_id': idx_set}

        # Create individual simulation cases for each datapoint in this set
        properties = ChemKED(os.path.join(data_path, dataset), skip_validation=skip_validation)
        simulations = create_simulations(dataset, properties)

        ignition_delays_exp = numpy.zeros(len(simulations))
        ignition_delays_sim = numpy.zeros(len(simulations))

        #############################################
        # Determine standard deviation of the dataset
        #############################################
        ign_delay = [case.ignition_delay.to('second').value.magnitude
                     if hasattr(case.ignition_delay, 'value')
                     else case.ignition_delay.to('second').magnitude
                     for case in properties.datapoints
                     ]

        # get variable that is changing across datapoints
        variable = get_changing_variable(properties.datapoints)
        # for ignition delay, use logarithm of values
        standard_dev = estimate_std_dev(variable, numpy.log(ign_delay))
        dataset_meta['standard deviation'] = float(standard_dev)

        #######################################################
        # Need to check if Ar or He in reactants but not model,
        # and if so skip this dataset (for now).
        #######################################################
        if ((any(['Ar' in spec for case in properties.datapoints
                  for spec in case.composition]
                  )
             and 'Ar' not in model_spec_key[model_name]
             ) or
            (any(['He' in spec for case in properties.datapoints
                  for spec in case.composition]
                  )
             and 'He' not in model_spec_key[model_name]
             )
            ):
            warnings.warn('Warning: Ar or He in dataset, but not in model. Skipping.',
                          RuntimeWarning
                          )
            error_func_sets[idx_set] = numpy.nan
            continue

        # Use available number of processors minus one,
        # or one process if single core.
        pool = multiprocessing.Pool(processes=num_threads)

        # setup all cases
        jobs = []
        for idx, sim in enumerate(simulations):
            # special treatment based on pressure for Princeton model (and others)

            if model_variant and model_name in model_variant:
                model_mod = ''
                if 'bath gases' in model_variant[model_name]:
                    # find any bath gases requiring special treatment
                    bath_gases = set(model_variant[model_name]['bath gases'])
                    gases = bath_gases.intersection(
                        set([c['species-name'] for c in sim.properties.composition])
                        )

                    # If only one bath gas present, use that. If multiple, use the
                    # predominant species. If none of the designated bath gases
                    # are present, just use the first one (shouldn't matter.)
                    if len(gases) > 1:
                        max_mole = 0.
                        sp = ''
                        for g in gases:
                            if float(sim.properties['composition'][g]) > max_mole:
                                sp = g
                    elif len(gases) == 1:
                        sp = gases.pop()
                    else:
                        # If no designated bath gas present, use any.
                        sp = bath_gases.pop()
                    model_mod += model_variant[model_name]['bath gases'][sp]

                if 'pressures' in model_variant[model_name]:
                    # pressure to atm
                    pres = sim.properties.pressure.to('atm').magnitude

                    # choose closest pressure
                    # better way to do this?
                    i = numpy.argmin(numpy.abs(numpy.array(
                        [float(n)
                         for n in list(model_variant[model_name]['pressures'])
                         ]
                        ) - pres))
                    pres = list(model_variant[model_name]['pressures'])[i]
                    model_mod += model_variant[model_name]['pressures'][pres]

                model_file = os.path.join(model_path, model_name + model_mod)
            else:
                model_file = os.path.join(model_path, model_name)

            jobs.append([sim, model_file, model_spec_key[model_name], results_path, restart])

        # run all cases
        jobs = tuple(jobs)
        results = pool.map(simulation_worker, jobs)

        # not adding more proceses, and ensure all finished
        pool.close()
        pool.join()

        dataset_meta['datapoints'] = []

        for idx, sim in enumerate(results):
            sim.process_results()

            if hasattr(sim.properties.ignition_delay, 'value'):
                ignition_delay = sim.properties.ignition_delay.value
            else:
                ignition_delay = sim.properties.ignition_delay

            if hasattr(ignition_delay, 'nominal_value'):
                ignition_delay = ignition_delay.nominal_value * units.second

            dataset_meta['datapoints'].append(
                {'experimental ignition delay': str(ignition_delay),
                 'simulated ignition delay': str(sim.meta['simulated-ignition-delay']),
                 'temperature': str(sim.properties.temperature),
                 'pressure': str(sim.properties.pressure),
                 'composition': [{'InChI': sim.properties.composition[spec].InChI,
                                  'species-name': sim.properties.composition[spec].species_name,
                                  'amount': str(sim.properties.composition[spec].amount.magnitude),
                                  } for spec in sim.properties.composition],
                 'composition type': sim.properties.composition_type,
                 })

            ignition_delays_exp[idx] = ignition_delay.magnitude
            ignition_delays_sim[idx] = sim.meta['simulated-ignition-delay'].magnitude

        # calculate error function for this dataset
        error_func = numpy.power(
            (numpy.log(ignition_delays_sim) -
             numpy.log(ignition_delays_exp)) / standard_dev, 2
             )
        error_func = numpy.nanmean(error_func)
        error_func_sets[idx_set] = error_func
        dataset_meta['error function'] = float(error_func)

        dev_func = (numpy.log(ignition_delays_sim) -
                    numpy.log(ignition_delays_exp)
                    ) / standard_dev
        dev_func = numpy.nanmean(dev_func)
        dev_func_sets[idx_set] = dev_func
        dataset_meta['absolute deviation'] = float(dev_func)

        output['datasets'].append(dataset_meta)

        if print_results:
            print('Done with ' + dataset)

    # Overall error function
    error_func = numpy.nanmean(error_func_sets)
    if print_results:
        print('overall error function: ' + repr(error_func))
        print('error standard deviation: ' + repr(numpy.nanstd(error_func_sets)))

    # Absolute deviation function
    abs_dev_func = numpy.nanmean(dev_func_sets)
    if print_results:
        print('absolute deviation function: ' + repr(abs_dev_func))

    output['average error function'] = float(error_func)
    output['error function standard deviation'] = float(numpy.nanstd(error_func_sets))
    output['average deviation function'] = float(abs_dev_func)

    # Write data to YAML file
    with open(splitext(basename(model_name))[0] + '-results.yaml', 'w') as f:
        yaml.dump(output, f)

    return output
