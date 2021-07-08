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

from pyked.chemked import ChemKED, SpeciesProfileDataPoint  
from pyked.chemked import Composition # Added import to resolve picking error?
# Local imports
from .utils import units
from .jsr_simulation import JSRSimulation
from .plotting import generate_plots_jsr

min_deviation = 0.10
"""float: minimum allowable standard deviation for experimental data"""

def create_simulations(dataset, properties,target_species_name):
    """Set up individual simulations for each ignition delay value.

    Parameters
    ----------
    dataset :

    properties : pyked.chemked.ChemKED
        ChemKED object with full set of experimental properties

    Returns
    -------
    simulations : listsim
        List of :class:`Simulation` objects for each simulation

    """

    simulations = []
    for case in properties.datapoints:
        for idx,temp in enumerate(case.temperature):
            sim_meta = {}
            # Common metadata
            sim_meta['data-file'] = dataset
            sim_meta['id'] = splitext(basename(dataset))[0] + '_' + str(idx)

            simulations.append(JSRSimulation(properties.experiment_type,
                                        properties.apparatus.kind,
                                        sim_meta,
                                        case,target_species_name
                                        )
                            )
    return simulations

def simulation_worker(sim_tuple):
    """Worker for multiprocessing of simulation cases.

    Parameters
    ----------s
    sim_tuple : tuple
        Contains Simulation object and other parameters needed to setup
        and run case.

    Returns
    -------
    sim : ``Simulation``
        Simulation case with calculated ignition delay.

    """
    sim, model_file, model_spec_key, path, restart = sim_tuple

    sim.setup_case(model_file, model_spec_key, path)
    sim.run_case(restart)
    concentration = sim.process_results()

    sim = JSRSimulation(sim.kind, sim.apparatus, sim.meta, sim.properties,sim.target_species_name)
    return sim,concentration


def estimate_std_dev(indep_variable, dep_variable):
    """

    Parameters
    ----------
    indep_variable : ndarray, list(float)
        Independent variable (e.g., temperature, pressure)
    dep_variable : ndarray, list(float)
        Dependent variable (e.g., species profile)

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



"Not sure this def is needed as only concentration/temperature changes? @ below"

def get_changing_variables(case,species_name):
    """Identify variable changing across multiple cases. #ToDo: Do it for multiple cases
    e.g. Inlet temperature, inlet composition and target species 

    Parameters
    ----------
    case : pyked.chemked.SpeciesProfileDataPoint
         SpeciesProfileDataPoint with experimental case data.

    Returns
    -------
    variables : tuple(list(float))
       Tuple of list of floats representing changing experimental variable.

    """

    inlet_composition = {}
    for k,v in case.inlet_composition.items():
        inlet_composition[k] = v.amount.magnitude.nominal_value
    target_species_profile = [quantity for quantity in case.outlet_composition[species_name].amount]
    inlet_temperature = [quantity for quantity in case.temperature]
    variables = [target_species_profile,
                inlet_temperature,
    ]
    
    return variables



"""thoughts: 
1. ideally inchi/species identifies are listed in yaml file/csv? so spec_keys_file may be unnecessary: Anthony
    But I think we need spec key file : Krishna
2."""

def evaluate_model(model_name, spec_keys_file, species_name,
                   dataset_file,
                   data_path='data', model_path='models',
                   results_path='results', model_variant_file=None,
                   num_threads=None, print_results=True, restart=False,
                   skip_validation=True, create_plots=True, plot_path='jsr_plots'):

    """Evaluates the species profile error of a model for a given dataset.

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
    create_plots : bool
        If ``True``, generates plots for all species in species key.
    plot_path : bool
        Local path for creating the plots pdf. Optional; default = 'jsr_plots'

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
        simulations = create_simulations(dataset, properties,target_species_name=species_name)

        species_profile_exp = numpy.zeros(len(simulations))
        species_profile_sim = numpy.zeros(len(simulations))

        #############################################
        # Determine standard deviation of the dataset and get variables
        # Krishna: not doing standard deviation for now 
        #############################################

        # get variable that is changing across datapoints
        variables = [get_changing_variables(dp,species_name=species_name) for dp in properties.datapoints]
        
        #standard_dev = estimate_std_dev(variable, numpy.log(species_profile))
        #dataset_meta['standard deviation'] = float(standard_dev)

        #######################################################
        # Need to check if Ar or He in reactants but not model,
        # and if so skip this dataset (for now).
        #######################################################
        """
        I don't think we need this for JSR simulations 
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
        """
        # Use available number of processors minus one,
        # or one process if single core.
        pool = multiprocessing.Pool(processes=num_threads)

        # setup all cases
        jobs = []
        for idx, sim in enumerate(simulations):
            model_file = os.path.join(model_path, model_name)
            jobs.append([sim, model_file, model_spec_key[model_name], results_path, restart])


        # run all cases
        """
        Deleting this for now because of weird picking error
        jobs = tuple(jobs)
        results = pool.map(simulation_worker, jobs)

        # not adding more proceses, and ensure all finished
        pool.close()
        pool.join()
        """
        results = []
        for job in jobs:
            results.append(simulation_worker(job))

        dataset_meta['datapoints'] = []
        expt_target_species_profiles = {}
        simulated_species_profiles  = []
        for idx, sim_tuple in enumerate(results):
            sim,concentration = sim_tuple
            

            expt_target_species_profile, inlet_temperature = get_changing_variables(properties.datapoints[0],species_name=species_name)
            #Only assumes you have one csv : Krishna
            dataset_meta['datapoints'].append(
                {'experimental species profile': str(expt_target_species_profile),
                 'simulated species profile': str(concentration),
                 'temperature': str(sim.properties.temperature),
                 'pressure': str(sim.properties.pressure),
                 })

            expt_target_species_profiles[str(idx)] = [quantity.magnitude for quantity in expt_target_species_profile]
            simulated_species_profiles.append(concentration)
            #assert (len(expt_target_species_profile)==len(sim.meta['simulated_species_profiles'])), "YOU DONE GOOFED UP SIMULATIONS"

        # calculate error function for this dataset
        experimental_trapz = numpy.trapz(inlet_temperature,expt_target_species_profile)
        print (simulated_species_profiles)
        simulated_trapz = numpy.trapz(inlet_temperature,simulated_species_profiles)
        if print_results:
            print ("Difference between AUC:{}".format(experimental_trapz-simulated_trapz))

    # Write data to YAML file
    with open(splitext(basename(model_name))[0] + '-results.yaml', 'w') as f:
        yaml.dump(output, f)

    # Generate species concentration plots
    if (create_plots):
        generate_plots_jsr(model_name, model_path, results_path, spec_keys_file, data_path, plot_path)

    return output