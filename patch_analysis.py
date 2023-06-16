
# install matplotlib, numpy, scipy, tkinter, pyabf

import tkinter as tk 
import pyabf
import matplotlib.pyplot as plt
import numpy as np 
from scipy.signal import find_peaks
from os import listdir
import pickle
import scipy.stats as stats
from scipy.stats import sem


## TO DO : Add abf.dataRate to infos (= frame per seconds)
freqs = {'cc' : 50000, 'vc' : 10000}
Vrest_values = {'23425' : -65, '23509' : -76}

sweeps_mb  = [0,1,2]
dict_user = {}
dict_user['file'] = {'label' : 'path of the file to analyse', 'var' : '', 'value' : ''}
dict_user['folder'] = {'label' : 'path of the folder', 'var' : '', 'value' : ''}
dict_user['index_channel'] = {'label' : 'number of channel to use (NB: write 0 for the first one etc)', 'var' : '', 'value' : ''}
dict_user['LJP'] = {'label' : 'liquid junction potential', 'var' : '', 'value' : '-14.5'}

dict_user['delay_cc'] = {'label' : 'delay due to the software (s)', 'var' : '', 'value' : '0.062'}
dict_user['step_start_cc'] = {'label' : 'time when the hyperpolarizing step starts (s)', 'var' : '', 'value' : '0.050'}
dict_user['step_stop_cc'] = {'label' : 'time when the hyperpolarizing step stops (s)', 'var' : '', 'value' : '0.250'}
dict_user['step_cc'] = {'label' : 'value of the hyperpolarizing step (pA)', 'var' : '', 'value' : '-50'}
dict_user['start_cc'] = {'label' : 'time when the current step starts (s)', 'var' : '', 'value' : '2.5'}
dict_user['stop_cc'] = {'label' : 'time when the current step stops (s)', 'var' : '', 'value' : '3.3'}
dict_user['min_cc'] = {'label' : 'first value of the current injected (pA)', 'var' : '', 'value' : '0'}
dict_user['cur_step_cc'] = {'label' : 'step between each current (pA)', 'var' : '', 'value' : '50'}
dict_user['threshold_PA'] = {'label' : 'minimum voltage value for action potential (mV)', 'var' : '', 'value' : '0'}

dict_user['delay_vc'] = {'label' : 'delay due to the software (s)', 'var' : '', 'value' : '0.078'}
dict_user['step_start_vc'] = {'label' : 'time when the hyperpolarizing step starts (s)', 'var' : '', 'value' : '0.020'}
dict_user['step_stop_vc'] = {'label' : 'time when the hyperpolarizing step stops (s)', 'var' : '', 'value' : '0.120'}
dict_user['step_vc'] = {'label' : 'value of the hyperpolarizing step (mV)', 'var' : '', 'value' : '-5'}
dict_user['start_vc'] = {'label' : 'time when the ramp starts (s)', 'var' : '', 'value' : '0.420'}
dict_user['stop_vc'] = {'label' : 'time when the ramp stops (s)', 'var' : '', 'value' : '3.420'}
dict_user['min_vc'] = {'label' : 'first value of the ramp (mV)', 'var' : '', 'value' : '-50'}
dict_user['max_vc'] = {'label' : 'last value of the ramp  (mV)', 'var' : '', 'value' : '55'}
dict_user['Vrest'] = {'label' : 'Resting membrane potential (mV)', 'var' : '', 'value' : '-65'}
dict_user['V_g1'] = {'label' : 'Potential for conductance ref 1 (mV)', 'var' : '', 'value' : '-100'}
dict_user['V_g2'] = {'label' : 'Potential for conductance ref 2 (mV)', 'var' : '', 'value' : '-25'}
dict_user['V_g_window'] = {'label' : 'Window around potential for conductance slope (mV)', 'var' : '', 'value' : '10'}


keys = {}
keys['vc'] = ['delay_vc', 'LJP', 'step_start_vc', 'step_stop_vc', 'step_vc', 'start_vc', 'stop_vc', 'min_vc', 'max_vc', 'Vrest', 'V_g1', 'V_g2', 'V_g_window']
keys['cc'] = ['delay_cc', 'LJP', 'step_start_cc', 'step_stop_cc', 'step_cc', 'start_cc', 'stop_cc', 'min_cc', 'cur_step_cc', 'threshold_PA']

# PLOT colors 
alpha_plots = 0.2
alpha_color = 0.4
edgecolor = 'black'
facecolor = 'white'
colors = {'baseline' : 'grey', 'terbinafine' : 'cadetblue', 'NPBA_10' : 'goldenrod', 'NPBA_20' : 'goldenrod','washout' : 'grey', 'terbinafine_100' : 'darkcyan'}
inverse_color = {'black': 'white', 'white' : 'black', 'grey':'black', 'goldenrod':'black', 'cadetblue':'black', 'darkcyan':'black'}

# PLOT elements sizes 
dot_size = 1 #cm
width_bp = 0.6 #%
lw = 1
lw_median = 2
tick_thick = 1
tick_len = 0.1 #cm
space_ax = 0.5
fig_len = 5 # cm 
whis = (10,90) # Percentile for whiskers 


# Text 
font = 'Arial'
fontsize = 11
folder_fig = r'W:\Analysis\patch\figures'
format_fig = 'png'

def all_sweeps(l, k, ignore = []) :
    ignore = [int(i) for i in ignore]
    return [i for i in l if i not in ignore]

def select_sweeps(l, k, ignore = []) : 
    k = [int(i) for i in k]
    ignore = [int(i) for i in ignore]
    return [i for i in k if i not in ignore and i in l]

def start_sweeps(l,k, ignore = []) : 
    k = int(k)
    ignore = [int(i) for i in ignore]
    if k < len(l) :
        l = l[k:]
    return [i for i in l if i not in ignore]

def stop_sweeps(l,k, ignore = []) : 
    k = int(k)
    ignore = [int(i) for i in ignore]
    if k < len(l) :
        l = l[:k]
    return [i for i in l if i not in ignore]


def read_abf_file(file, list_sweeps = None, print_info = False, ask_user = False, funct_sweeps = None, echo = False, membrane = False) : 
    abf = pyabf.ABF(r'{}'.format(file))
    if print_info :
        print(file);print(abf)
    if abf.channelCount != 1 : 
        dict_obj = make_choices(dict_user, text_keys = ['index_channel'], checkbox_keys=[])
        channel_ID = int(dict_obj['index_channel']['value'])
    else : 
        channel_ID = 0
    infos = {}
    n_sweeps = abf.sweepCount
    if list_sweeps is None : 
        if funct_sweeps is None : 
            list_sweeps = list(range(n_sweeps))
        else : 
            list_sweeps = funct_sweeps[0](list(range(n_sweeps)), funct_sweeps[1], funct_sweeps[2])
    else : 
        list_sweeps = [sweep for sweep in list_sweeps if sweep < n_sweeps]
    
    if membrane :
        list_sweeps = list_sweeps[:3]
    
    n_sweeps = len(list_sweeps)

    
    
    infos['n_sweeps'] = n_sweeps
    infos['sample_rate'] = abf.sampleRate
    infos['x_label'] = abf.sweepLabelX
    infos['y_label'] = abf.sweepLabelY

    if 'Current' in infos['y_label'] : 
        infos['mode'] = 'vc'
    elif 'Potential' in infos['y_label'] : 
        infos['mode'] = 'cc'
    

    if echo : 
        for key in infos.keys() :
            print(f'{key} : {infos[key]}')
        print(f'Sweeps selected : {list_sweeps}')

    text_keys = keys[infos['mode']]
    
    if ask_user : 
        dict_obj = make_choices(dict_user, text_keys= text_keys, checkbox_keys=[])
    else  :
        dict_obj = dict_user 
        if infos['mode'] == 'vc' : 
            for key in Vrest_values.keys() : 
                if key in file : 
                    dict_obj['Vrest']['value'] = Vrest_values[key]

    for key in text_keys : 
        infos[key] = float(dict_obj[key]['value'])
        
    data_matrix = [[] for _ in range (n_sweeps)]
    time_matrix = [[] for _ in range (n_sweeps)]
    for k, sweep in enumerate(list_sweeps) :
        abf.setSweep(sweepNumber=sweep, channel=channel_ID)
        data_matrix[k] = abf.sweepY
        time_matrix[k] = abf.sweepX
        if k == 0 : 
            n_points = len(time_matrix[k])
        elif len(time_matrix[k]) != n_points :
            print('Error in the number of sweeps')
    
    return [time_matrix, data_matrix, infos]

def divide_data(X, Y, infos, average = False) :
    timing = {}
    timing['step'] = [infos[f'step_start_{infos["mode"]}'] + infos[f'delay_{infos["mode"]}'],infos[f'step_stop_{infos["mode"]}'] + infos[f'delay_{infos["mode"]}']]
    timing['baseline'] = [0,timing['step'][0]]
    timing['exp'] = [infos[f'start_{infos["mode"]}'] + infos[f'delay_{infos["mode"]}'], infos[f'stop_{infos["mode"]}'] + infos[f'delay_{infos["mode"]}']]
    data = {}
    time = {}
    if average :
        X = np.mean(X, axis = 0)
        Y = np.mean(Y, axis = 0)
    for key in timing.keys() : 
        if average :
            data[key] = [Y[i] for i in range (len(Y)) if X[i] >= timing[key][0] and X[i] <= timing[key][1]]
            time[key] = [X[i] for i in range (len(X)) if X[i] >= timing[key][0] and X[i] <= timing[key][1]]
        else :
            data[key] = [[] for _ in range (len(Y))]
            time[key] = [[] for _ in range (len(X))]
            for sweep in range(len(Y)) :
                data[key][sweep] = [Y[sweep][i] for i in range (len(Y[sweep])) if X[sweep][i] >= timing[key][0] and X[sweep][i] <= timing[key][1]]
                time[key][sweep] = [X[sweep][i] for i in range (len(X[sweep])) if X[sweep][i] >= timing[key][0] and X[sweep][i] <= timing[key][1]]
    
    return data, time 

def membrane_properties(X, Y, infos, show_ref_resist = False, print_info = False) : 
    
    data, time = divide_data(X, Y, infos, average= True)
    
    if show_ref_resist : 
        f, ax = start_fig()
        ax.plot(data['step'], color = 'grey')
        ax = set_ax_parameters(ax, nticks = None, show_xticks=True)
        ax.set_xticks([int(len(data['step'])/2)], ['Point'])
        plt.show()
    
    if infos['mode'] == 'cc' : 
        Vrest = np.mean(data['baseline'])
        infos['Vrest'] = Vrest
        V = Vrest-data['step'][int(len(data['step'])/2)]
        I = infos['step_cc']

    elif infos['mode'] == 'vc' : 
        I = data['step'][int(len(data['step'])/2)] - np.mean(data['baseline'])
        V = infos['step_vc']
    
    infos['Rm'] = ((abs(V*1e-3))/abs((I*1e-12))) # in ohm
    
    if print_info :
        print(f'Membrane resistance = {infos["Rm"]/1e6} MOhm')

    
    ## Add possibility for Tau and Cm
    return infos


def plot_all(X,Y, ax = None, color = 'black', plot_protocole = False, show = True) :
    if ax is None : 
        if plot_protocole :
            f, axes = start_fig(nrows = 2)
            protocole = [0 for _ in range (len(X))]
            axes[1].plot(X,protocole, color = 'black')
            axes[1] = set_ax_parameters(axes[1], just_plot=True)
        else :
            f, ax = start_fig()
            axes = [ax] 
    else : 
        axes = [ax]
    axes[0].plot(X,Y, color = color)
    axes[0] = set_ax_parameters(axes[0], just_plot=True)
    if show :
        plt.show()
    else : 
        return axes 
    
def ramp_results(X, Y, infos, plot_curve = False, plot_protocole = False, plot_g_ref = False) : 

    data, time = divide_data(X, Y, infos, average= True)
    infos['max_vc'] += infos['Vrest']
    infos['min_vc'] += infos['Vrest']
    voltage_step = (infos['max_vc'] - infos['min_vc'])/len(time['exp'])
    voltage = [infos['min_vc'] + i*voltage_step + infos['LJP'] for i in range (len(time['exp']))] # correct for LJP

    for point in range (len(data['exp'])) : 
        if abs(data['exp'][point]) < 0.05 :
            infos['Vrev'] = voltage[point]
            point_i = point

    if plot_curve :
        if plot_protocole :
            f, axes = start_fig(nrows = 2)
            axes[1].plot(time['exp'],voltage, color = 'black')
            axes[1] = set_ax_parameters(axes[1], just_plot=True)
        else :
            f, ax = start_fig()
            axes = [ax]
        axes[0].plot(voltage, data['exp'], color = 'grey')
        axes[0] = set_ax_parameters(axes[0], nticks = None, show_xticks=True)
        axes[0].scatter(voltage[point_i], data['exp'][point_i], color = 'red')

    infos['voltage'] = voltage
    infos['current'] = data['exp']

    for k in [1,2] :
        infos[f'V_g{k}'] += infos['LJP']
        g_x = [voltage[i] for i in range (len(data['exp'])) if voltage[i] >= infos[f'V_g{k}'] - infos['V_g_window']/2 and voltage[i] <= infos[f'V_g{k}'] + infos['V_g_window']/2]
        g_y = [data['exp'][i] for i in range (len(data['exp'])) if voltage[i] >= infos[f'V_g{k}'] - infos['V_g_window']/2 and voltage[i] <= infos[f'V_g{k}'] + infos['V_g_window']/2]
        for i in range (len(data['exp'])) :
            if abs(voltage[i]-infos[f'V_g{k}']) <= 2*voltage_step : 
                infos[f'I_{k}'] = data['exp'][i] 
        coeffs = np.polyfit(g_x, g_y, 1)
        if plot_g_ref : 
            y = [coeffs[0]*x + coeffs[1] for x in g_x]
            axes[0].plot(g_x, y, color = 'blue', lw = 2)

        infos[f'g_{k}'] = coeffs[0]

    infos[f'I_rect'] = abs(infos['I_1']/infos['I_2'])
    
    if plot_curve : 
        plt.show()
    
    return infos

def current_freq(X, Y, infos, plot_freq = False) : 
    data, time = divide_data(X, Y, infos, average= False)

    current = [0]*len(data['exp'])
    freq = [0]*len(data['exp'])
    for sweep in range (len(data['exp'])) :
        current[sweep] = infos['min_cc'] + sweep*infos['cur_step_cc']
        index, _ = find_peaks(data['exp'][sweep], height = infos['threshold_PA'])
        freq[sweep] = len(index)/(time['exp'][sweep][-1] - time['exp'][sweep][0])
    
    infos['frequency'] = freq
    infos['current'] = current 

    if plot_freq :
        f, ax = start_fig()
        ax.plot(current, freq, color = 'grey')
        ax = set_ax_parameters(ax, nticks = None, show_xticks=True)
        plt.show() 
    return infos 

def input_file() :
    dict_start = make_choices(dict_user, text_keys=['file'], checkbox_keys=[])
    file = r'{}'.format(dict_start['file']['value'])
    return file 

def choose_files_from_folder(folder = '', title = '') :
    '''
    TO DO 
    '''


    if folder == '' : 
         # Ask the user what is the folder with the recordings 
        dict_obj = make_choices(dict_user, text_keys = ['folder'], checkbox_keys = [], title = title)
        # Save the file path in r format to avoid problems with '\'
        folder = r'{}'.format(dict_obj['folder']['value'])
    
    files = listdir(folder)
    files = [r'{}'.format(file) for file in files if '.abf' in file]

    dict_obj = {}
    for file in files :
        dict_obj[file] = {'label' : file, 'value' : 0, 'var' : ''}
    
    dict_obj = make_choices(dict_obj, text_keys = [], checkbox_keys = files, title = title)

    paths = []
    keys_sup = ['list', 'start', 'stop', 'except']
    for file in files :
        if dict_obj[file]['value'] == 1 :
            path = folder + '\\' + file 
            paths.append(r'{}'.format(path))
            list_keys = []
            dict_obj[f'{path}_list'] = {'label' : 'List of sweeps', 'value' : '', 'var' : ''}
            dict_obj[f'{path}_start'] = {'label' : 'or sweeps from..', 'value' : '', 'var' : ''}
            dict_obj[f'{path}_stop'] = {'label' : 'or sweeps until..', 'value' : '', 'var' : ''}
            dict_obj[f'{path}_except'] = {'label' : 'Sweeps to ignore', 'value' : '', 'var' : ''}
            for key in keys_sup : 
                list_keys.append(f'{path}_{key}')
            dict_obj = make_choices(dict_obj, text_keys = list_keys, checkbox_keys = [], title = file)

    sweeps_func = {} 
    for path in paths : 
        except_sweeps = dict_obj[f'{path}_except']['value'].replace(' ', '').split(',')
        if except_sweeps == [''] :
            except_sweeps = []
        list_sweeps = dict_obj[f'{path}_list']['value'].replace(' ', '').split(',')
        start_sweep = dict_obj[f'{path}_start']['value']
        stop_sweep = dict_obj[f'{path}_stop']['value']
        if list_sweeps != [''] : 
            sweeps_func[path] = [select_sweeps, list_sweeps, except_sweeps]
        elif start_sweep != '' : 
            sweeps_func[path] = [start_sweeps, start_sweep, except_sweeps]
        elif stop_sweep != '' : 
            sweeps_func[path] = [stop_sweeps, stop_sweep, except_sweeps]
        else :
            sweeps_func[path] = [all_sweeps, None, except_sweeps]
    
    return paths, sweeps_func


def cm2inch(x):
    return x/2.54

def cm2pts(x) : 
    return x*28.35

def pts2inch(x) : 
    return x/72


def start_fig(ncols = 1, nrows = 1, height_ratios = None, width_ratios  = None, pad = 0.8, figsize = None) :
    if figsize is None :
        figsize = (ncols * (cm2inch(fig_len)+pts2inch(3*pad)), nrows * (cm2inch(fig_len) + pts2inch(pad)))
    f, ax = plt.subplots(ncols = ncols, nrows= nrows, figsize = figsize, gridspec_kw={'height_ratios':height_ratios, 'width_ratios' : width_ratios})
    f.tight_layout(pad=pad, w_pad=3*pad, h_pad=pad)
    return f, ax 

def set_ax_parameters(ax, nticks = None, yticks = None, just_plot = False, xlabel = None, ylabel = None, xticks = None, show_xticks = False, show_yticks = True) :
    '''
    For all my plots, I need to delete the top and right lines on the plot, use a certain tick length and width, and no label on the ticks (but need to see their value to put them in Illustrator later)
    Input: ax to change
    Output: changed ax 
    '''

    ax.spines[['top','right']].set_visible(False) # remove the black lines 

    if xticks is not None :
        show_xticks = True

    if show_xticks :
        if xticks is None : 
            xticks = ax.get_xticks()
            if nticks is not None : 
                xstep = (max(xticks)-min(xticks))/nticks
                xticks = [min(xticks) + i*xstep for i in range (nticks)]
                print(xticks)
                xticks = rounding(xticks)
                print(xticks)
                ax.set_xticks(xticks[1:], xticks[1:], font = font, fontsize = fontsize)
            else : 
                ax.set_xticks(xticks[1:-1], xticks[1:-1], font = font, fontsize = fontsize)
        else  :
            ax.set_xticks(xticks, font = font, fontsize = fontsize)


        
        #ax.tick_params(axis="x", length = 0)
        ax.tick_params(axis="x",direction="in", length = cm2pts(tick_len), width = tick_thick) # ticks going in the plot at the right size 

    else : 
        ax.set_xticks([]) # For the x axis no ticks at all


    if show_yticks :
        if yticks is None :
            yticks = ax.get_yticks()
            if nticks is not None : 
                ystep = (max(yticks)-min(yticks))/nticks
                yticks = [min(yticks) + i*ystep for i in range (nticks)]
                yticks = rounding(yticks)
                ax.set_yticks(yticks[1:], yticks[1:], font = font, fontsize = fontsize)
            else : 
                ax.set_yticks(yticks[1:-1], yticks[1:-1], font = font, fontsize = fontsize)
        else : 
            ax.set_yticks(yticks, yticks, font = font, fontsize = fontsize)
        ax.tick_params(axis="y",direction="in", length = cm2pts(tick_len), width = tick_thick) # ticks going in the plot at the right size 
    else :
        ax.set_yticks([])

    if ylabel is not None :
        ax.set_ylabel(ylabel, font = font, fontsize = fontsize)

    if xlabel is not None :
        ax.set_xlabel(xlabel, font = font, fontsize = fontsize)
    
    if just_plot : 
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['bottom', 'left']].set_visible(False) # remove the black lines 
    
    return (ax)

def rounding(list) : 
    '''
    TO DO 
    '''

    def power_of_x(x) : 
        power = 1
        while x >= 10:
            x /= 10
            power += 1
        return power


    def find_factor(power) :
        if power < 2 :
            factor = 1 
        elif power == 2 :
            factor = 5
        elif power == 3 : 
            factor = 25
        else : 
            factor = int((10 ** (power-2))/4)

        return factor 
    
    rangelist = abs(max(list) - min(list))
    
    if rangelist < 2 : 
        return [round(x,2) for x in list]
    else :
        new_list = []
        for x in list : 
            if abs(x) < 1 :
                new_list.append(round(x,2))
            elif abs(x) < 2 :
                new_list.append(round(x,1))
            else : 
                power = power_of_x(x)
                f = find_factor(power)
                possibilities = [f*(x//f), f*(x//f + 1)]
                new_list.append(int(possibilities[np.argmin([abs(x-p) for p in possibilities])]))

        return new_list

def make_choices(dict_object: dict, text_keys: list, checkbox_keys: list, title = '') : 
    '''
    This functions allows to show a window where the user can put information either in text or in check boxes 

    Input 
    dict_object (dict): dictionnary where each key is a variable of interest with the different subkeys: 'label', 'var', 'value' 
            'label': information to show to the user
            'var': where we store temporary variables 
            'value': the input of the user 

    text_keys (list of str): list of the keys we use when the function is called (for text entries)

    checkbox_keys (list of str): list of the keys we use when the function is called (for checkboxes)

    Output
    dict_object (dict): the dictionnary after modifications 
    '''

    def div (x, d = 10) : 
        m = x //d
        k = x - d*(m)
        return (m,k)

    def save_entry_fields():
        '''
        Get the user input (string) and store it in the dictionnary 
        '''
        # Go through all the defined keys for text entries 
        for key in text_keys :
            # Save the user input in a dictionnary 
            dict_object[key]['value'] = dict_object[key]['var'].get()

    def save_entry_fields_box():
        '''
        Get the user input (box checked or not) and store it in the dictionnary 
        '''
        # Go through all the defined keys for check_box entries 
        for key in checkbox_keys :
            # Save the user input in a dictionnary 
            dict_object[key]['value'] = dict_object[key]['var'].get()


    # Open a new window 
    master = tk.Tk()
    master.title(title)
    
    # Place the window in the center of the screen 
    master.eval('tk::PlaceWindow . center')

    # For each key related to text entries
    for i,key in enumerate(text_keys) :

        # Show text with the label defined in the dictionnary for this key 
        tk.Label(master, text=dict_object[key]['label']).grid(row=i)

        # Save the tk enty object in the dictionnary 
        dict_object[key]['var'] = tk.Entry(master)
        dict_object[key]['var'].insert(0, dict_object[key]['value'])

        # Place this entry object at the right place in the window (each entry is a row and they are all in the column 1)
        dict_object[key]['var'].grid(row=i, column=1)
    
    col = 1 
    row = len(text_keys) + 1  
    
    
    # For each key related to check boxes entries
    for i,key in enumerate(checkbox_keys) :
        
        col, row = div(i)
        col += 2
        row += 1

        # Create a tk variable for integers 
        dict_object[key]['var']= tk.IntVar()

        # Create a check box with the label defined in the dictionnary for this key 
        c = tk.Checkbutton(master, text=dict_object[key]['label'],variable=dict_object[key]['var'], onvalue=1, offvalue=0, command=save_entry_fields_box)

        # Place this checkbox at the right place in the window (each one is a row and they are all in the column 1)
        c.grid(row=row, column = col)
                     

    # Create a button for saving choices (text entries)    
    tk.Button(master, text='Save your choices', command=save_entry_fields).grid(row=row+1, column=col, sticky=tk.W, pady=4)

    tk.mainloop()
    return dict_object 


def save_dict(var, pathfile) : 
    '''
    saves variable var in a picke file based on the path chosen 
    NB : do not forget to put .pkl at the end of the path, if just a name like file.pkl will be put in the working directory
    '''
    with open(pathfile, 'wb') as fp:
        pickle.dump(var, fp)

def read_dict(pathfile) :
    '''
    Returns variable from a picke file based on the path of this file 
    '''
    with open(pathfile, 'rb') as fp:
        var = pickle.load(fp)
    return var 

from random import randint 

def make_boxplot(y, experiments_keys, namefig = 'plot', saving = False, draw_line = False, stars = [], limy = None, broken = None, fig = None, ax = None, show = True, yticks = None, ylabel = None, xticks = None) : 
    '''
    Creates boxplot based on values in y and a list of keys corresponding (y and experiments_keys have the same len)
    y : list of lists each list contains the values for each condition 
    experiments_keys : list of names of the conditions (with associated colors in colors_plot in the parameters_file)
    namefig: the name of the figure for saving 
    Put saving = True so the figure is saved 

    TO DO adjust the height of the stat bar so no overlap with dots 
    '''
    
    # Get the right colors 
    try :
        colors_plot = [colors[exp] for exp in experiments_keys]
        median_colors = [inverse_color[color] for color in colors_plot]
    except :
        colors_plot = ['white' for _ in experiments_keys]
        median_colors = ['black' for _ in experiments_keys]
    
    y_box = [np.array(yi) for yi in y]
    y_box = [yi[~np.isnan(yi)].tolist() for yi in y_box]
    stars = statistics(y_box, show_ns = False, print_stats = True)
    
    for i in range (len(y)) : 
        print(f'Median : {np.nanmedian(y[i])}. Std : {np.nanstd(y[i])}, Mean: {np.nanmean(y[i])}, SEM: {sem(y[i])}')
    if ax is None :
        _, ax = start_fig(1, 1)
    
    bp = ax.boxplot(y_box, whis = whis, widths=width_bp, showfliers=False, patch_artist=True)
    
    if limy is None :
        ylim = ax.get_ylim()
        maxi = np.nanmax([np.nanmax(yi) for yi in y])
        mini = np.nanmin([np.nanmin(yi) for yi in y])
        if not (np.isnan(mini)) and not (np.isinf(mini)) : 
            limy = [min(mini,ylim[0])*0.9, max(ylim[1], maxi)*1.1]

        else :
            limy = ylim

    if limy == [None, None] :
        limy = ax.get_ylim()

    ax.set_ylim(limy)
    
    x = [[i + 1.1 + width_bp/2 for _ in range(len(y[i]))] for i in range (len(experiments_keys))]
    x = [space_dot_boxplot(xi,yi, limy) for xi,yi in zip(x,y)]

    if draw_line: 
        for j in range (len(y[0])) :
            xline = []
            yline = []
            for i in range (len(experiments_keys)) :
                xline.append(x[i][j])
                yline.append(y[i][j])
            ax.plot(xline,yline, color = 'grey', alpha = alpha_plots)
    
    ax = set_ax_parameters(ax, yticks = yticks, ylabel = ylabel, xticks = xticks)

    if stars is not None : 
        ax = add_stats(ax, stars, limy)
    
    points = []
    for xi, yi,c in zip(x,y, colors_plot) :
        points.append(ax.scatter(xi, yi, color = c, edgecolors=edgecolor, s = cm2pts(dot_size)))


    for patch, color, median, median_color in zip(bp['boxes'], colors_plot, bp['medians'], median_colors):
        patch.set_facecolor(color)
        patch.set_linewidth(lw)
        median.set_color(median_color)
        median.set_linewidth(lw_median)

    if saving :
        plt.tight_layout()
        plt.savefig(f'{folder_fig}\{namefig}.{format_fig}', transparent = True)
    if show : 
        plt.show()
    else : 
        return ax 

def space_dot_boxplot(x,y,limy) :
    '''
    Messy function to put some horizontal space between dots that are too close together in scattering

    Input: x = list of coordinates to space if need 
           y = list of fixed coordinates
           limy = [min y axis, max y axis]
    '''
    toignore = []
    try : 
        distanceval = abs(limy[1]-limy[0])/20
    except :
        distanceval = 0.1

    for i in range (len(y)) :
        if i not in toignore : # This is a new value we encounter 
            c = [i]
            for j in range (i+1, len(y)) : # Create a list of all values too close to the value i of interest
                if abs(y[i]-y[j]) < distanceval :
                    c.append(j)
                    toignore.append(j) # We don't need to look at these values again later on 
            
            if len(c) == 2 : # 2 values close together, one a bit on the left, one on the right 
                x[c[0]] -= 0.025
                x[c[1]] += 0.025
            if len(c) == 3 :  # 3 values close together, leaves the middle one, and put one a bit on the left, one on the right 
                x[c[0]] -= 0.05
                x[c[2]] += 0.05
            if len(c) == 4 :  # 4 values close together, 2 a bit on the left, 2 on the right (with a bit space between each)
                x[c[0]] -= 0.05
                x[c[1]] -= 0.03
                x[c[2]] += 0.02
                x[c[3]] += 0.05
            if len(c) > 4 : # More than 5 values ?! Well just put them by turn one one the left one on the right at random distances
                f = -1
                for j in range (len(c)) :
                    x[c[j]] += f*(randint(2,8))/100
                f = 1 if f == -1 else -1 
    return x 

def statistics(y, show_ns = False, print_stats = False) :
    '''
    From a list of results y (with each element of y are the same size), returns statistics 
    Input: y list of lists of same len to statistically compare
          test a function for comparing the different groups 
    Ouput: stars list of significant differences for each [index of reference group in y, index of compared group in y, stars to put depending on significance level]
    '''
    def test_before_ttest(A,B, pval = 0.05) :
        sampling_difference = np.array(A) - np.array(B)
        _,p1 = stats.shapiro(sampling_difference)
        _, p2 = stats.levene(A,B,center= 'mean')
        if p1 > pval and p2 > pval :
            return True
        else :
            return False

    stars = [] 

    # Go through lists to compare 
    for i in range (len(y)) :
        # Compare one list to all the next ones 
        for j in range (i+1, len(y)) : 
            emptytest1 = [k for k in y[i] if k not in [None, 0, np.nan]]
            emptytest2 = [k for k in y[j] if k not in [None, 0, np.nan]]
            if len(emptytest1) > 2 and len(emptytest2) > 2 : 
                if test_before_ttest(y[i],y[j]) :
                    test = 'ttest'
                    pval = stats.ttest_ind(y[i], y[j]).pvalue # Find the pvalue
                else :
                    test = 'wilcoxon'
                    pval = stats.wilcoxon(y[i], y[j]).pvalue # Find the pvalue

                if print_stats : 
                    print(i,j,pval, test)
                # Assign number of stars depending on the pvalue 
                if pval < 0.001 :
                    stars.append([i,j,'***', pval, test]) 
                elif pval < 0.01 :
                    stars.append([i,j,'**', pval, test])
                elif pval < 0.05 :
                    stars.append([i,j,'*', pval, test])
                elif show_ns : 
                    stars.append([i,j,'ns', pval, test])
                
    return stars 


def add_stats(ax, stars, limy) : 
    up = limy[1]
    y_range = limy[1]-limy[0]
    for i, star in enumerate(stars):
        # Columns corresponding to the datasets of interest
        x1 = star[0] + 1
        x2 = star[1] + 1
        # What level is this bar among the bars above the plot?
        level = len(stars) - i
        # Plot the bar
        bar_height = (y_range * 0.2 * level) + limy[1]
        bar_tips = bar_height - (y_range * 0.02)
        ax.plot(
                    [x1, x1, x2, x2],
                    [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
                )
        text_height = bar_height + (y_range * 0.01)
        ax.text((x1 + x2) * 0.5, text_height, star[2], ha='center', va='bottom', c='k')
        up = max(up,text_height)

    ax.set_ylim([limy[0], up])
    return (ax)
