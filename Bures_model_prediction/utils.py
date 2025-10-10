import numpy as np
import pandas as pd
import pickle
from random import gauss
import matplotlib.pylab as plt
import matplotlib.image as img

from tensorflow.keras import Model

from tensorflow.keras.layers import Dense, Input, LSTM, concatenate
from tensorflow.keras.utils import Sequence 

pd.options.mode.chained_assignment = None  # default='warn'
        
def load_data_with_error (path = 'Data/', string = 'M1_M20_train_val_test_set'):
    '''
    Load the data files for training, validation and testing

    Parameters
    ----------
    path : TYPE, string
        Folder where the data is saved. The default is 'Data/'.
    string : TYPE, string
        Name of the data files. The default is ''.

    Returns
    -------
    data : list of numpy arrays
        'x1_train', 'x2_train', 'y_train', 'x1_val', 'x2_val', 'y_val', 'x1_test', 'x2_test', 'y_test' as numpy arrays

    '''
    filenames = ['x1_train', 'x2_train', 'y_train', 'x1_val', 'x2_val', 'y_val', 'x1_test', 'x2_test', 'y_test']
    data = []
    
    for i,file_name in enumerate(filenames):
        with open(path + file_name+'_' +string +'.pkl', 'rb') as f:
            data.append(pickle.load(f))
    return data


def create_model_lstm(input1_shape, input2_shape, output_shape):
    '''
    Create the model's graph in Keras

    Parameters
    ----------
    input1_shape : shape of the model's input data (using Keras conventions).
    input2_shape : shape of the model's input data (using Keras conventions).
    output_shape : shape of the model's output data (using Keras conventions).

    Returns
    -------
    model : keras model instance.

    '''

    
    initial_concentrations = Input(shape= input1_shape)
    kinetics = Input(shape = input2_shape)

    c = Dense(200, activation = 'relu')(initial_concentrations)
    c = Model(inputs = initial_concentrations, outputs = c)
    
    

    k = LSTM(200, return_sequences = True)(kinetics)
    k = LSTM(200)(k)
    k = Model(inputs = kinetics, outputs = k)
    
    combined = concatenate([c.output, k.output])
    
    pred = Dense(200, activation = 'relu', kernel_initializer = 'he_uniform')(combined)
    
    pred = Dense(output_shape[1], activation = 'softmax', name = 'Dense_5')(pred)

    model = Model(inputs = [c.input, k.input], outputs=pred)
    
    return model   


class BatchGenerator(Sequence):
    
    '''
    Data generator for data augmentation during training.
    X : list of x1 and x2 inputs
    y : labels as one-hot vectors
    tps : list of time point values for the time point generator
    error_range : list of %values for standard deviation of Gaussian error to be added to concentration data
    seed_value : seed for the random generator
    batch_size : batch size output
    Generates data for Keras, this is a faster version of random_tp, where
        the same random points are taken across the whole batch, rather than different points
        for each example within the batch. It runs 6 times faster.
        n_runs =  how many kinetic runs per example
    '''
    
    def __init__(self, X, y, tps, error_range, seed_value = 1, batch_size=1024, shuffle=True):
        'Initialization'

        self.x1, self.x2 = X
        self.y = y
        self.tps = tps
        self.error_range = error_range
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_batches = self.__len__()
        self.n_runs = 4
        self.species = [1,2]
        self.error_species = len(self.species)
        self.error_dict = {key:[] for key in error_range}

        #sets a fixed randomness for reproducibility
        np.random.seed(seed_value)
        self.randomstate = np.random.default_rng(seed_value)

        self.on_epoch_end()      
        
        examples,timepoints,curves = self.x2.shape
        columns = curves // self.n_runs

        self.index_species = []
        for i in range(self.n_runs):
            t_species = [a+(i*columns) for a in self.species]
            self.index_species = self.index_species + t_species
        
      
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y)/self.batch_size))

    def __getitem__(self, index):
        return self.__data_generation(index)

    def on_epoch_end(self):
        'Shuffles indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        for error in self.error_range:
            if error != 0:
                preu = np.array([gauss(0,error/100) for i in range(self.batch_size * np.max(self.tps) * self.error_species * self.n_runs)])
                self.error_dict[error] = np.reshape(preu,(self.batch_size,np.max(self.tps),self.error_species * self.n_runs))

    def __data_generation_old(self, index):
        x2b = self.trim_x2(self.x2[self.batch_size*index: self.batch_size*(index+1)], self.tps, index)
        x2b = self.introduce_error(x2b)
        
        self.newx1 = self.x1[self.batch_size*index: self.batch_size*(index+1)]
        self.newy = self.y[self.batch_size*index: self.batch_size*(index+1)]
        
        return [self.newx1, x2b], self.newy

    def __data_generation(self, index):
        #does putting all without assigning variables solve the memory leak problem? Does it increase speed?
        x1 = self.x1[self.batch_size*index: self.batch_size*(index+1)]
        x2 = self.introduce_error(self.trim_x2(self.x2[self.batch_size*index: self.batch_size*(index+1)], self.tps, index))
        y = self.y[self.batch_size*index: self.batch_size*(index+1)]
        x = [x1,x2]

        return x, y, [None]
    
    
    def trim_x2(self, original_x2, tps,index):
        '''
        Takes a x2 set and reduces the number of tps to that specified (plus 0,0)
        This is a reduced version of trim, where the same points are taken for all samples in the batch
        '''
        original_tps = original_x2.shape[1]
        
        n_tps = self.randomstate.choice(tps)

        #takes n_tps out of the total length-1, adds the 0 tp and sorts them in order
        index = np.sort(np.append(self.randomstate.choice(original_tps-1,n_tps,replace = False)+1, [0]))

        return original_x2[:,index].copy()

    def introduce_error(self, x2):
        '''
        Take an lstm formatted x2 and apply error to the concentration data

        n_runs = how many kinetic runs there are for each example
        species = position of species to introduce error, in the first kinetic run
        keep_zero = must be true if we have 0,0 points in the dataset
        error_type = 'absolute' will add/substract an absolute value calculated from 'error' as standard deviation

        '''
        examples,timepoints,curves = x2.shape
        
        error = self.randomstate.choice(self.error_range)
        if error != 0:
            x2[:,1:,self.index_species] = x2[:,1:,self.index_species]+self.error_dict[error][:,0:timepoints-1]
        
        return x2
 


def one_hot(array):
    '''
    Transform mechanism labels into one-hot vectors

    Parameters
    ----------
    array : numpy array with labels

    Returns
    -------
    numpy array with one-hot vectors.

    '''
    max_val = array.max()+1

    return np.eye(max_val)[array[:,0]]

def predict_from_file(model, data_file, columns = ['Time','A','P','catT'], A0 = False, seed = 1):
    '''
    Takes four kinetic runs: 3 of them at =! concentrations of [cat]0 and a fourth with a same excess-type experiment; and predicts the mechanism.
    The kinetics must be provided in the corresponding csv template: it must be in four columns in this order: Time, A, P, CatT (only time zero is needed for catT).
    Time zero with initial concentrations must be the first entry
    
    '''
    def predict(x1,x2):
        predictions_array = model.predict([x1,x2])[0]
        
        return predictions_array
    
    def csv_to_example(df,A0):
        #Load files
        kruns = [0]*samples
        for i in range(samples):
            kruns[i] = df.iloc[:,(i*columns_per_sample):(columns_per_sample)+(i*columns_per_sample)].copy()
            kruns[i].columns = columns
            kruns[i].dropna(axis = 0, how = 'all', inplace = True)
        #check all runs are the same length
        same_length = True
        min_len = len(kruns[0])
        for run in kruns:
            if len(run) != min_len:
                same_length = False
                if len(run) < min_len:
                    min_len = len(run)
                
        if not same_length:
            #We must cut all runs to the min_length. We will do this randomly
            print(f'Reducing all traces to {min_len} timepoints')
            randomstate = np.random.default_rng(seed)
            
            for i in range(samples):
                if len(kruns[i]) > min_len:
                    index = np.sort(np.append(randomstate.choice(len(kruns[i])-1, min_len-1, replace = False)+1,[0]))
                    kruns[i] = kruns[i].iloc[index]    
        
        # AI accepted format before normalization
        x1 = np.array([[kruns[i].catT.iloc[0] for i in range(samples)] ])
        x2 = np.concatenate([kruns[i][columns[0:-1]].values for i in range(samples)], axis = 1)
        x2 = np.expand_dims(x2, axis = 0)
        original_data = (x1,x2)

        #Normalize time, A, Pand catT
        longest_run = np.max([kruns[i].Time.iloc[-1] for i in range(samples)])
        if not A0:
            #Limiting reagent must be second column in columns list
            A = columns[1]
            A0 = kruns[0][A].iloc[0]
            if A0 == 0:
                print('Please check your data: [substrate] must be the second column of each run, or [substrate]0 must be provided with the command --S0')
                exit()
            print('Initial concentration of starting material,',A,':', A0)

        for i in range(samples):
            kruns[i].Time = kruns[i].Time / longest_run
            for specie in columns[1:]:
                kruns[i][specie] = kruns[i][specie] / A0
            #print(kruns[i])

    

        #create AI accepted format
        x1 = np.array([[kruns[i].catT.iloc[0] for i in range(samples)] ])
        x2 = np.concatenate([kruns[i][columns[0:-1]].values for i in range(samples)], axis = 1)
        x2 = np.expand_dims(x2, axis = 0)
        return x1, x2, (original_data, A0)
    
    #load files
    print('Predicting mechanism for data in ', data_file)
    if A0:
        print('With [SM]0:',A0)
    df = pd.read_csv(data_file, sep = '\t')
    #let's remove rogue columns and rows from the txt file
    df.dropna(axis = 0, how = 'all', inplace = True)
    df.dropna(axis = 1, how = 'all', inplace = True)
    #print(df)

    columns_per_sample = len(columns)
    samples = int(len(df.columns) /  columns_per_sample)
    
    #transform into model input
    x1, x2, original_data = csv_to_example(df,A0)


    #predict 
    predictions = predict(x1,x2)
    
    return predictions, original_data, samples

def generate_predicted_grouping(predictions):
    
    threshold = 0.99

    index = np.argsort(predictions)
    prob =0
    grouping = []
    for j in index[::-1]:
        prob += predictions[j]
        grouping.append(j)
        if prob >= threshold:
            break
    
    return grouping
    
def grouping_index_to_names(grouping):

    mechanism_list = ['M'+str(i) for i in range(1,21)]

    grouping_names = [mechanism_list[i] for i in grouping]
    
    return grouping_names



def plot_kinetics_mech(x_list, grouping_names, save_name = None, columns = 3, s = 30, A0 = 1, show = True, overlay = False, labels = None):
    '''
    plots a number of kinetics
    x_list = lstm formatted (x1,x2)
    columns = how many plots per row in x_list; ie how many kinetic runs per sample

    '''

    (x1,x2) = x_list
    rows = len(x1)
    curves = x2.shape[2] // columns
    if not overlay:
        figure = plt.figure(figsize = (4 * columns, 7 * rows))

    i = 1
    for row in range(rows):
        for column in range(columns):
            plt.subplot(2, columns, i)
            if not overlay:
                # If it is the same-excess experiment do not calculate the cat loading
                if column == 3:
                    title = r'[Cat]$_{0}$ = ' + str(round(x1[row, column],5))
                else:
                    percentage = ' (' +str(round(x1[row, column]/A0*100, 3))+' mol%)'
                    title = r'[Cat]$_{0}$ = ' + str(round(x1[row, column],5)) + percentage
                plt.title(title)
            if type(labels) == list:
                label = labels
            else:
                label = None
            for j in range(1,curves):
                if type(labels) == list:
                    label = labels[j-1]
                else:
                    label = None
                plt.scatter(x2[row,:,column*curves], x2[row,:,(j)+column*curves], s, label = label)
            if not overlay:
                plt.xlim(left = 0, right = max(x2[row,:,column*curves])*1.1)
                plt.ylim(bottom = 0)
            i += 1                
    plt.tight_layout()
    mechs = []
    for i,name in enumerate(grouping_names):
        mechs.append(img.imread('Images/'+name+'.jpg'))
    length = len(mechs)
    for i,mech in enumerate(mechs):
        plt.subplot(2,length,length +i+1)
        plt.axis('off')
        plt.imshow(mech)
    if save_name:
        plt.savefig(f'all_data_plot/prediction-{save_name}.png')
    else:
        plt.savefig('prediction.png')


model_columns = {
    'M1_20_model.h5':           ['Time','S','P','catT'],
    'M1_20_model_P_noXS.h5':    ['Time','P','catT'],
    'M1_20_model_S_noXS_01to5.h5':    ['Time','S','catT'],
    'M1_20_model_S_noXS.h5':          ['Time','S','catT'],
    'M1_20_model_P_noPXS_01to5.h5':  ['Time','S','P','catT'],
    'M1_20_model_P_noPXS.h5':         ['Time','S','P','catT'],
    'M1_20_model_S_noPXS.h5':         ['Time','S','catT'],
    'M1_20_model_S_noPXS_01to5.h5':         ['Time','S','catT']
        }

    
