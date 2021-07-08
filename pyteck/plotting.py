import pandas as pd
import h5py
import matplotlib.pyplot as plt
import yaml
import cantera as ct
import os

from matplotlib.backends.backend_pdf import PdfPages


def generate_plots_jsr(model_name, model_path, results_path, spec_keys_file, data_path, plot_path):

    """Generates plots of steady-state concentration over temperature for each species in the species key.

    Parameters
    ----------
    model_name : str
        Chemical kinetic model filename
    model_path : str
        Local path for model file. Optional; default = 'models'
    results_path : str
        Local path for creating results files. Optional; default = 'results'
    spec_keys_file : str
        Name of YAML file identifying important species
    data_path : str
        Local path for data files. Optional; default = 'data'
    plot_path : bool
        Local path for creating the plots pdf. Optional; default = 'jsr_plots'

    """

    sol = ct.Solution(os.path.join(model_path, model_name))
    h5_list = os.listdir(results_path)
    experimental_files = os.listdir(data_path)

    spec_names_model = (sol.species_names)

    for file in experimental_files:
        if os.path.splitext(file)[-1] == ".yaml":   # Any none .yaml files are skipped over
            if os.path.exists(os.path.join(data_path, file)):
                print(f"Loading {file}")
                with open(os.path.join(data_path, file), 'r') as f:
                    data = yaml.load(f, Loader=yaml.SafeLoader)

            else:
                raise OSError(f"Couldn't find {os.path.join(data_path,file)}")
        else:
            print(f"Ignoring none .yaml file {file}")
            continue

        if os.path.exists(spec_keys_file):
            with open(spec_keys_file, 'r') as k:
                key = yaml.load(k, Loader=yaml.SafeLoader)
        else:
            raise OSError(f"Couldn't find {spec_keys_file}")

        species = data['datapoints'][0]['outlet-composition']['species']
        # experimental file should contain the path to the corresponding csv file
        csvfile = data['datapoints'][0]['csvfile']

        if os.path.exists(csvfile):
            exp = pd.read_csv(csvfile)
            print(f"Loading {csvfile}")
        elif os.path.exists(os.path.join('data', csvfile)):
            exp = pd.read_csv(os.path.join('data', csvfile))
            print(f"Loading {os.path.join('data', csvfile)}")
        else:
            raise OSError(f"Couldn't find {csvfile}")

        with PdfPages(os.path.join(plot_path, 'jsr_plots') + '-' + model_name + '.pdf') as plot_pdf:
            for sp in species:
                name_in_data = sp['species-name']

                if name_in_data in key[model_name].keys():

                    name_in_model = key[model_name][sp['species-name']]
                    print('Plotting concentration for ' + name_in_model)

                    temps = []
                    concs = []

                    # get the position in which this species is listed in the model
                    i = 0
                    for name in spec_names_model:
                        if(name == name_in_model):
                            break
                        else:
                            i = i + 1
                    # iterate through all results files (each for a single temp)
                    for h5 in h5_list:
                        f = h5py.File(os.path.join(results_path, h5), 'r')
                        dset = f['simulation']

                        temp = dset[1][1]
                        conc = (dset[-1][4][i])

                        concs.append(conc)
                        temps.append(temp)

                        f.close()

                    temps, concs = zip(*sorted(zip(temps, concs)))  # sort both lists

                    plt.figure()
                    plt.plot(temps, concs, linestyle='solid')
                    plt.title(sp['species-name'] + ' concentration')

                    temps = exp['Temperature']
                    concs = exp[sp['species-name']]

                    plt.scatter(temps, concs)
                    plt.legend(['simulated', 'experimental'])
                    plt.xlabel('Temperature (K)')
                    plt.ylabel('Mole Fraction')

                    plot_pdf.savefig()
                    plt.close()
