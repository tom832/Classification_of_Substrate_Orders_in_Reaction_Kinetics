import os
import json
#Remove tensorflow loading messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
#Use only CPU
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import argparse
import matplotlib.pylab as plt

from utils import predict_from_file, generate_predicted_grouping, grouping_index_to_names, model_columns, plot_kinetics_mech

from tensorflow.keras.models import load_model

import os
os.system('color')


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help = 'data file; should be a txt with TAB delimitations')
    parser.add_argument('--model', help = 'Trained model file; should be a h5 file. Default is M1_20_model.h5')
    parser.add_argument('--S0', type = float, help = 'Initial concentration of starting material for models with only [P] data')
    parser.add_argument('--columns', type = list, help = 'List of columns used in each sample in data set. Eg [''Time'', ''S'', ''P'', ''catT'']\nIf present, the limiting reagent must be the column after Time. If not present, its initial concentration must be provided with the parameter --S0')
    parser.add_argument('--plot', type = bool, help = 'If True it will plot the normalized data fed to the model, for visual inspection. Default: True')
    parser.add_argument('--plot_name', type = str)
    
    args = parser.parse_args()
    
    
    data_file = args.filename
        
    if args.plot:
        plot = args.plot
    else:
        plot = True
    
    if args.plot_name:
        plot_name = args.plot_name
    else:
        plot_name = None
        
    if args.model:
        model_file = args.model
    else:
        model_file = 'Data/M1_20_model.h5'
        
    model_name = os.path.basename(model_file)

    if args.S0:
        S0 = args.S0
    else:
        S0 = None
    
    try:
        model = load_model(model_file, compile = False)
    # catch exceptions and print error message
    except Exception as e:
        print(e)
        print('ERROR: the model could not be found at',model_file)
        print('\tAvailable pretrained models:', end = ' ')
        for name in model_columns.keys():
            print(name,end = ', ')
        exit()
    
    if model_name in model_columns:
        columns = model_columns[model_name]
    else:
        if args.columns:
            columns = args.columns
        else:
            print('\n\nERROR: You must provide a list of columns used in this model.\n\n')
            exit()
    
    # print('\nUsing model:', model_file)
    # print('This model requires:',columns, 'per sample')
    predictions, original_data, samples = predict_from_file(model, data_file, columns = columns, A0 = S0)
    print(type(predictions))
    if plot_name:
        with open(f'all_data_output/{plot_name}.json', 'w') as f:
            json.dump(predictions.tolist(), f, indent=2)
    else:
        with open('pred.json', 'w') as f:
            json.dump(predictions, f, indent=2)
    grouping = generate_predicted_grouping(predictions)
    grouping.sort()
    grouping_names = grouping_index_to_names(grouping)
    
    
    if len(grouping_names) == 1:
        print('Kinetic data is consistent with mechanism\033[1m',grouping_names[0], "\033[0m")
    else:
        print('Kinetic data is consistent with mechanisms\033[1m', *grouping_names, "\033[0m", sep = ' ')
    
    if plot:
        print('Plotting results')
        try:
            plot_kinetics_mech(original_data[0], grouping_names, columns = samples, A0 = original_data[1], save_name=plot_name)
            plt.show()
        except:
            print('There was an error when plotting. Double-check that Images folder is included in the folder in which you are running this script.')
    
if __name__ == '__main__':
    # try:
    main()
    # except:
    #     print('\nThere was an ERROR running this prediction. Double-check: \n-that you have selected the right trained model for the shape of your data, and provided the correct path to the model file.')

        
