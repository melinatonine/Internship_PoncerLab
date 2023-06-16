import numpy as np

peak_types = ['interictal', 'MUA']
min_freq = {'interictal' : 0.05, 'MUA' : 0.1}
max_freq = {'interictal' : 1, 'MUA' : 15}


infos_interest = {'unit' : {'index' : 6, 'val' : b'V'}, 'exponent' : {'index' : 7, 'val' : -12}, 'step' : {'index' : 8, 'val' : 0}, 
                  'freq' : {'index' : 9, 'val' : 100}, 'factor' : {'index' : 10, 'val' : 59605}}


proportion_slices_IILDs = {'SH-2022-002' : {'yes':[5,6,7,9], 'no' : [3,4,8]}, 'SH-2022-004' : {'yes' : [5,7,8], 'no' : [1,2,3,4,6,9,10]}, 'SH-2022-005' : {'yes' : [2,3,5,6], 'no' : [1,4,7,8,9,10]}, 'SH-2023-000' : {'yes' : [2,3], 'no' : [1,4]}, 'SH-2023-001' : {'yes': [3,4,5,6,8,9], 'no' : [10,7]}}

factor_amp = infos_interest['factor']['val']*10**(infos_interest['exponent']['val'])*1e6 # convert to µV 

# Definition threhsold interictal peaks
def thresholding_interictal(signal) :
    return 7*np.std(signal)

# Definition threshold MUA
def thresholding_MUA(signal) :
    return 8*np.std(signal)


# Normalisation functions 
def normalization(signal) :
    sig_n = (signal - np.mean(signal))/np.std(signal)
    # We use abs(x) to take into account negative peaks 
    return abs(sig_n)

## Time info
freqs = 10000 # acquisition frequency /s
time_max = None
after_time = 3.5*60*freqs 
duration = 3.5*60*freqs
period = int(60*freqs) # sampling for computations in frames for plots 
timepower = 0.150*freqs # time before and after peak for computing its power (in frames)


# Parameters for analysis 
# NB the threshold can be a float/int or a function of the signal that returns a float/int
param = {}
param['interictal'] = {'filter_order' : 3, 'filter_type' : 'bandpass', 'filter_freq' : [1,40], 'inter_event' : 0.250*freqs, 'peak_duration' : None, 'threshold' : thresholding_interictal, 
                        'color_plot' : 'darkred', 'color_peak' : 'coral'}
param['MUA'] = {'filter_order' : 5, 'filter_type' : 'bandpass', 'filter_freq' : [300,3000], 'inter_event' : 0.001*freqs, 'peak_duration' : None, 'threshold' : thresholding_MUA, 
                    'color_plot' : 'mediumblue', 'color_peak' : 'powderblue'} 
denoising_param = {'wavelet' : 'db4', 'mode' : 'hard', 'wavelet_levels' : 5}

# Types of data saved in the excel file 
data_types = 'raw', 'filtered', 'normalized'

# Files
folder = r'W:\Analysis\MEA\task3\\'
folder_fig = folder + r'figures\\'
peaks_file = folder  + r'task3_peaks.xlsx'
peaks_folder = folder  + r'peaks_folder\\'
selected_peaks_file = folder + r'task3_peaks_thresholded.xlsx'
positions_file = folder + r'task3_positions.xlsx'
info_file = folder + r'task3_infos.xlsx'
results_file = folder + r'task3_results.xlsx'
format_slices = ['jpg', 'png', 'bmp']


# relevant info for excel info file 
infos_keys = ['path', 'channels', 'condition', 'slice', 'ID', 'layout']
start_row = 2

# relevant info for excel results file 
excel_columns_interest = range(2,9)
n_conditions = 3
N_best = 4

# relevant info for excel positions file 
center_channels_options = {'dense' : ['F6', 'G6', 'F7', 'G7'], 'sparse' : ['F5', 'G5', 'F6', 'G6']}
list_channels_all = {'sparse': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'K10', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10'], 'dense': ['A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'K10', 'K11', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']}

# PLOT figure 
format_fig = 'svg'
fig_size = (10,10) 

# PLOT colors 
alpha_plots = 0.3
alpha_color = 0.4
edgecolor = 'black'
facecolor = 'white'
colors = {'baseline' : 'grey', 'terbinafine' : 'cadetblue', 'NPBA_10' : 'goldenrod', 'NPBA_20' : 'goldenrod','washout' : 'grey', 'terbinafine_100' : 'darkcyan', 'task3 mut' : 'black', 'HS' : 'white'}
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

# Text 
font = 'Arial'
fontsize = 12

# specific 
text_space = 10 # parameters for layout decision 
border = 100 # same 
steps_color = 50 # N colors in colorbar for heatmap 
whis = (10,90) # Percentile for whiskers 


# For bursts
max_time_interval_in_burst = 3*freqs
min_peaks_in_burst = 10 

# GUI 
gui_dictionnary = {}

gui_dictionnary['interictal'] = {'label' : 'Interictal', 'var' : [],'value' : 0}
gui_dictionnary['MUA'] = {'label' : 'MUA', 'var' : [],'value' : 0}

gui_dictionnary['channels'] = {'label' : 'List of channels', 'var' : [],'value' : ''}







