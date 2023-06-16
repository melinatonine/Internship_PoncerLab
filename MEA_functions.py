
# Info from computer 
from os.path import basename, normpath, dirname
from os import listdir
from pathlib import Path 


# Packages from signal processing 
from scipy import signal
import h5py 
from scipy import signal as signal_func 
from skimage.restoration import denoise_wavelet 
from scipy.stats import sem

# For gui/plots
import tkinter as tk 
import matplotlib.pyplot as plt 
import matplotlib.widgets as wdg
import matplotlib.cm as clm 
import matplotlib.patches as pt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

# For text
import re
import string

# Math
import numpy as np 
from random import randint
from math import isnan 
import scipy.stats as stats
from math import log10


# Reading data 
import pandas as pd 
import openpyxl as op 
import h5py
import pickle

from parameters_file_BRAF import * 



## Getting started 

def fill_info_file(return_row = False) : 
    '''
    This function reads the info file from parameters_file and let us fill the information with a GUI and put them after the last row 
    No input, no output
    '''
    paths = choose_files_from_folder() # We give a folder and there will be a list of files from which we can choose the one we want 
    dico = {}
    for n,path in enumerate(paths) :

        # Open the info file 
        wb = op.load_workbook(info_file) 
        sheet = wb['conditions'] # Go to the right sheet 
        row = sheet.max_row + 1 # Start writing after last row to avoid overwrite 

        date_time = date_time_from_path(path)

        # Using the GUI, the user can enter the right information for each relevant info (see parameters_file: info_keys)
        if n == 0 : 
            for key in infos_keys[1:] :
                dico[key] = {'label' : f'For file {date_time}, enter {key}', 'var': [], 'value':''}
            dico['around'] = {'label' : f'For file {date_time}, look at neighboring channels?', 'var': [], 'value': 0}

        else : 
            for key in infos_keys[1:] :
                dico[key]['label'] = f'For file {date_time}, enter {key}'
            dico['around']['label'] = f'For file {date_time}, look at neighboring channels?'

        dico = make_choices(dico, text_keys = infos_keys[1:], checkbox_keys=['around'])

        # The channels specified in here are only the channel of interest seen during the recording but the channels around might be interesting too 
        channels = dico['channels']['value']
        if dico['around']['value'] :
            channels = channels_around(channels)
        channels = ', '.join(channels)

        # We fill the information in the excel file 
        column = 1 
        sheet.cell(row=row, column=column).value = path
        for key in infos_keys[1:] : 
            column += 1 
            if key == 'channels' :
                sheet.cell(row=row, column=column).value = channels 
            else : 
                sheet.cell(row=row, column=column).value = dico[key]['value']

        # Done, the file is saved 
        wb.save(info_file)
    
    if return_row :
        return row-n


def show_image_tag_channels(date_time: string, channels: list) : 
    '''
    Input: date_time the date and the time of the recording of interest 
    Channels : list of channels to tag 
    '''

    def position_channels(channels: list, ref: list, distances:dict) : 
        '''
        Inputs: list of channels 
                ref : [name of ref channel, its position]
                distances {letter : the distance (x,y) between a channel and the next channel with the same number but different letter,
                        number : the distance (x,y) between a channel and the next channel with the same letter but different number}

        Outputs: x_channels with the x of all the channels in the same order as input channels (list of floats)
                y_channels with the y of all the channels in the same order as input channels (list of floats)

        '''
        # List of letters in the alphabet without the I (no I in the channels' names)
        alphabet = list(string.ascii_uppercase)
        i = alphabet.index('I')
        del alphabet[i]

        # The reference channel is the upper left of the center channels (indice 0)
        ref_letter_index = alphabet.index(ref[0][0]) # We find its position in the alphabet (B would be second etc)
        ref_number = int(ref[0][1:]) 

        # Initialize variables for output 
        x_channels = []
        y_channels = []

        # For each channel 
        for channel in channels :
            diff_letter = alphabet.index(channel[0]) - ref_letter_index # There are X letters between this channel and the ref 
            diff_number = int(channel[1:]) - ref_number # There are X numbers between this channel and the ref 
            pos = ref[1] + distances['letter']*diff_letter + distances['number']*diff_number # We can compute its position 

            x_channels.append(pos[0]) # add x to the list 
            y_channels.append(pos[1]) # add y to the list 

        return x_channels, y_channels

    
    # Get the path of the image based on the date and time of the experiment 
    image = get_image_from_info(date_time)
    print(image)

    # What we ask the user to tag 
    center_channels = 'center up left', 'center up right', 'center bottom left', 'center bottom right'

    # Convert the image into an array of RGB values 
    img = np.asarray(Image.open(image))

    # Plot the image 
    fig, ax = plt.subplots(figsize=(20,20))
    fig.set_facecolor('black')
    ax.imshow(img)
    plt.ion()

    # We add borders around the figure where we will put text later on 
    xlim = np.array(ax.get_xlim()) + [-border,+border]
    ax.set_xlim(xlim)

    ylim = np.array(ax.get_ylim()) + [-border,+border]
    ax.set_ylim(ylim)

    # The position of the text in the image 
    xtextleft = xlim[0] - text_space
    xtextright = xlim[1] + text_space
    ytextup = ylim[1] + text_space

    # In parameters file we give two types of layout and each has different center channels, the keys are the names of the layout 
    keys = list(center_channels_options.keys())


    # We ask the user to click on the first channel of interest 
    plt.text(xtextleft, ytextup, f'Click on {center_channels[0]}', size=15, 
            ha="center", va="center",
            bbox=dict(boxstyle="round",
                    ec=edgecolor,
                    fc=facecolor,
                    ))
    

    
    # Need to create a class to save the values with which we interact using a GUI 
    class define_grid :
        points = [] 
        i = 0
        x_c = {}
        y_c = {}
        tokeep = None
        pause = True 

        def mouse_event(self, event):
            # We get the x,y coordinates of the mouse when clicking 
            self.x = event.xdata
            self.y = event.ydata


            # If we have not yet clicked on all the 4 channels needed:
            if self.i < len(center_channels) : 

                
                plt.scatter(self.x, self.y, color = 'white') # We put a white dot on the last clicked channel 
                self.points.append(np.array([self.x,self.y])) # We save the coordinates of this channel 
                self.i += 1 # We count that we have saved one more channel 

                if self.i < len(center_channels) : # Do we still have more channels to click on? 
                    # If yes, we put on the screen what we should click on next 
                    plt.text(xtextleft,ytextup,f'Click on {center_channels[self.i]}', size=15, 
                    ha="center", va="center",
                    bbox=dict(boxstyle="round",
                            ec= edgecolor,
                            fc= facecolor,
                            ))

                else : 
                    # If there are no more channels to click on, we let the user decide which layout they want to try 

                    # Layout 1 click on the left side of the screen 
                    plt.text(xtextleft, ytextup,f'Click here for {keys[0]} layout', size=15, 
                        ha="center", va="center",
                        bbox=dict(boxstyle="round",
                                ec=edgecolor,
                                fc=facecolor,
                                ))
                    
                    # Layout 2 click on the right side of the screen 
                    plt.text(xtextright, ytextup, f'Click here for {keys[1]} layout', size=15, 
                        ha="center", va="center",
                        bbox=dict(boxstyle="round",
                                ec=edgecolor,
                                fc=facecolor,
                                ))

                    # We compute the distances between the channels we clicked on, whats the distance between two channels with different letters? Numbers?
                    # Since we have four points, we have two distances for each case and take the mean of the two
                    self.distances = {'letter' : np.mean([self.points[1]-self.points[0], self.points[3]-self.points[2]], axis=0), 'number' : np.mean([self.points[2]-self.points[0], self.points[3]-self.points[1]], axis=0)}

                    # For each layout, we find out where would be the channels on the image using the function 'position_channels' (go to this function for more info)
                    for c in range (2) : 
                        self.ref = [center_channels_options[keys[c]][0], self.points[0]]
                        xi,yi = position_channels(channels[keys[c]], self.ref, self.distances)
                        self.x_c[keys[c]] = xi
                        self.y_c[keys[c]] = yi 

                    self.i += 1
                
               
            else :
                if abs(self.x - xtextleft) < abs(self.x - xtextright) :
                    show = 0
                    hide = 1
                
                # If our click is closer to the second option, we will show the second layout (1) and hide the first layout (0)
                else :
                    show = 1
                    hide = 0 

                # We put first a blank n_channels on the channels of the layout to hide 
                for xi,yi in zip(self.x_c[keys[hide]], self.y_c[keys[hide]]) : 
                    plt.text(xi,yi,'     ', size=12, 
                        ha="center", va="center",
                        bbox=dict(boxstyle="round",
                                ec='black',
                                fc='black',
                                ))
                    
                # Then we can show the right layout 
                for xi,yi,chan in zip(self.x_c[keys[show]], self.y_c[keys[show]], channels[keys[show]]) : 
                    plt.text(xi,yi,chan, size=12, 
                        ha="center", va="center",
                        bbox=dict(boxstyle="round",
                                ec='white',
                                fc='white',
                                ))
                
                # We have to remember which layout was the last clicked on  (it will be the one saved for later)
                self.tokeep = keys[show]

            # We have to show what changes we did 
            fig.canvas.draw_idle()


        def on_close(self, event):
            self.pause = False 
            # When the window is closed, the positions corresponding to the right layout are saved      
            add_sheet(positions_file, date_time) # We add a sheet for this file if needed

            # We open the file at the right sheet and put the headers 
            wb = op.load_workbook(positions_file)
            sheet = wb[date_time]
            row = 1 
            sheet.cell(row=row, column=1).value = 'channel'
            sheet.cell(row=row, column=2).value = 'x'
            sheet.cell(row=row, column=3).value = 'y'
            row += 1

            # For each channel, we put its x,y coordinates from the right layout (tokeep)
            for ci, xi, yi in zip(channels[self.tokeep], self.x_c[self.tokeep], self.y_c[self.tokeep]) : 
                sheet.cell(row=row, column=1).value = ci
                sheet.cell(row=row, column=2).value = xi
                sheet.cell(row=row, column=3).value = yi
                row += 1

            # We save the file 
            wb.save(positions_file)
            print(f'positions of {image} saved in {positions_file}')

            # We also put the type of layout in the info file at the right row in the last column 
            wb = op.load_workbook(info_file)
            sheet = wb['conditions']
            dt = date_time.replace('_','T')
            col_layout = infos_keys.index('layout') + 1
            for row in range (2,sheet.max_row + 1) : 
                path = sheet.cell(row=row, column=1).value
                if path is not None and dt in path :
                    sheet.cell(row=row, column=col_layout).value = self.tokeep
            
            # We save the file 
            wb.save(info_file)
            print(f'Layout of {image} saved in {info_file}')
    
         
    # The class needs to be called to be used 
    grid = define_grid()


    # Activate the functions depending on click 
    fig.canvas.mpl_connect('button_press_event', grid.mouse_event)
    fig.canvas.mpl_connect('close_event', grid.on_close)

    while grid.pause :
        plt.pause(0.1)
        
    plt.show()
    return
    

# Analysing the data 

def raw_signals_from_file(path:string, adding_sheet = False) :
    '''
    This functions is used to read a h5 file (exported with Data Manager)

    Inputs:
    file (str): path of the h5 file 

    Output:
    raw (array): matrix with the raw data. Shape is number_channels * time_points (duration in seconds*acquisition frequency)
    channels: list of channels in this recording 
    '''

    date_time = date_time_from_path(path)

    if adding_sheet : 
        # This takes 10 seconds so avoid if not necessary (use only for save_events_excel)
        add_sheet(peaks_file, date_time)

    # Get data from h5 file 
    data = h5py.File(path)

    # Find the right stream
    k = 0 
    while True :
        try :
            data = data['Data']['Recording_0']['AnalogStream'][f'Stream_{k}']
            break
        except :
            k += 1

    # Get the channels names
    all_channels = data['InfoChannel'] 
    c_index = 4 # the channel name is always at index 4 
    
    channels = [c[c_index].decode("utf-8") for c in all_channels] # convert bytes to string 
    
    # We get the data from all channels 
    raw = data['ChannelData']

    return raw, channels


class preprocessing :
    def __init__(self, raw:any, peak_types:list, denoise = None, filt = None, fact_threshold = None) :
        '''
        This class creates an object that is called 'self' within the class. The object will contain the raw data, filtered signal and normalized signal
        
        Inputs :
        raw (array): raw data from ONE channel

        Output :
        self (object): object with the raw data, (denoised signal), filtered signal and normalized signal.
        '''
        self.raw = raw
        self.peak_types = peak_types
        self.duration_s = len(self.raw)/freqs

        self.signal = {'filtered' : {}, 'normalized' : {}}
        self.threshold = {}

        if denoise :
            denoised = denoise_wavelet(self.raw, wavelet = denoising_param['wavelet'], mode = denoising_param['mode'], wavelet_levels = denoising_param['wavelet_levels'])
            denoised = denoised * (2**31 - 1) # convert int32 to float 

        # We go through the peak types (interictal and MUA) - They need different filtering 
        for peak_type in self.peak_types : 

            if filt is not None : 
                filter_type, filter_freq = filt[peak_type]
                b,a = signal.butter(param[peak_type]['filter_order'], filter_freq, btype = filter_type, fs = freqs)
            else : 
                # Filtering with defined parameters 
                b,a = signal.butter(param[peak_type]['filter_order'], param[peak_type]['filter_freq'], btype = param[peak_type]['filter_type'], fs = freqs)
                

            # Normalization of the filtered data: abs((X - mean(X))/std(X))
            if denoise : 
                self.signal['filtered'][peak_type] = signal.filtfilt(b, a, denoised) 
            else :
                self.signal['filtered'][peak_type] = signal.filtfilt(b, a, self.raw) 

            self.signal['normalized'][peak_type] = normalization(self.signal['filtered'][peak_type])
            
            # Apply thresholding function to find the right thresholds for this data set 
            if callable(param[peak_type]['threshold']):
                if fact_threshold is not None :
                    self.threshold[peak_type] = fact_threshold[peak_type]*param[peak_type]['threshold'](self.signal['normalized'][peak_type])
                else :
                    self.threshold[peak_type] = param[peak_type]['threshold'](self.signal['normalized'][peak_type])
             # Get threshold from parameter 
            else : 
                self.threshold[peak_type] = param[peak_type]['threshold']


class find_events :
    def __init__(self, data: object) : 
        '''
        This class creates an object that is called 'self' within the class. The object will contain the peaks' heights and time points + the mean amplitude of the peaks and their frequency

        Inputs:
        data (object): object created in preprocessing where we can find the raw data of one channel, but also the normalized data, etc
        type (str): 'interictal' or 'MUA' 
        threshold (int): a minimum value for peak detection (in mV)

        Output:
        self (object): object with the peaks' heights and time points + the mean amplitude of the peaks and their frequency
        '''

        self.total_frames = len(data.raw)
        self.thresholds = data.threshold
        self.parameters = param

        # Create empty dictionnaries to store information for each peak type 
        self.frame_index = {}
        self.amplitude = {}
        self.power = {}

        # We go through the peak types (interictal and MUA)
        for peak_type in data.peak_types : 
            self.amplitude[peak_type] = {}
            # Use the scipy function 'find_peaks' with defined parameters 
            self.frame_index[peak_type], properties = signal.find_peaks(data.signal['normalized'][peak_type], height = data.threshold[peak_type], distance = param[peak_type]['inter_event'], width = param[peak_type]['peak_duration'])

            # if at least one peak was found, find amplitude of each peak in the raw data and the normalized data 
            if len(self.frame_index[peak_type]) > 0 :
                self.amplitude[peak_type]['raw'] = [data.raw[index] for index in self.frame_index[peak_type]] # in mV 
                self.amplitude[peak_type]['normalized'] = properties['peak_heights']
                self.amplitude[peak_type]['filtered'] = [data.signal['filtered'][peak_type][index] for index in self.frame_index[peak_type]]
                self.power[peak_type] = [np.nanmean(data.raw[int(index-timepower):int(index+timepower)]**2) for index in self.frame_index[peak_type]]

            # No peak found, all variables = 0 
            else : 
                self.amplitude[peak_type]['raw'] = 0 # in mV 
                self.amplitude[peak_type]['filtered'] = 0 # in mV 
                self.amplitude[peak_type]['normalized'] = 0 # in mV 
                self.power[peak_type] = 0 # in mV
        
        


def raw_data_all_channels(path, frames, to_save = True, to_return = False) :
    '''
    This function is used to return (or save in a picke file) the raw data from all the channels between frames[0] and frames[1]
    Input
    path: path of the h5 file where the data is 
    frames: list with the [frame start, frame stop]
    to save: True/False; if True will save to a pickle file in the figure folder named with the date_time of the file and 'raw_channels' 
    to return: True/False; if True will return a dictionary with raw[number][letter] will contain the raw data between the frames of interest 
    '''

    # Need all the letters in order without the letter i (bc no 'i' in the nomenclature of MEA channels)
    alphabet = list(string.ascii_uppercase)
    i = alphabet.index('I')
    del alphabet[i]

    # Get the raw data and the list of channels 
    raw, channels_list = raw_signals_from_file(path)
    channels = ', '.join(channels_list)

    # Finds all the letters and numbers in the channels names 
    numbers = re.findall(r'\d+',channels)
    numbers = [int(number) for number in numbers]
    letters = re.findall(r'[a-zA-Z]',channels)
    min_letter = alphabet.index(min(letters))
    max_letter = alphabet.index(max(letters)) 

    # Puts the data in a dictionnary 
    raw_channels = {}
    for n in range (min(numbers), max(numbers) + 1) : 
        raw_channels[n] = {}
        for letter in range (min_letter, max_letter + 1) :
            raw_channels[n][alphabet[letter]] = 0
            channel = f'{alphabet[letter]}{n}'
            if channel in channels_list :
                channel_index = channels_list.index(channel)
                raw_channels[n][alphabet[letter]] = raw[channel_index][frames[0]:frames[1]]
            print(f'Done for {channel}', end = '\r')
    
    # If decided to save will save to a pickle file 
    if to_save : 
        date_time = date_time_from_path(path)
        save_dict(raw_channels, f'{folder_fig}\\{date_time}_raw_channels.pkl')
    
    # If decided to return will return the dict object 
    if to_return : 
        return raw_channels
    

# Getting / manipulating peaks 

def get_peaks_from_pkl(channel, date_time, selected_peaks = False) : 
    '''
    Pickle files contains the 'peaks' object with all the information about the peaks found for a specific recording 
    I decided to name them as '_date_time_channel.pkl' and put them in the peaks folder so they can be found easily
    Here we only take the frames when there were peaks, the amplitude of the peaks (in all 3 data types, raw filtered normalized), and the power of the peaks
    Inputs :
    channel : the channel of interest
    date_time : the date and time of the recording 
    selected_peaks : if we need to take the pickle object with only a sample of the peaks 
    Outputs : 
    frame_index : frames with peaks for both peak types 
    amplitude of the peaks in 3 data types 
    power of the peaks 
    '''

    peaks = []

    if selected_peaks :

        try : 
            peaks = read_dict(f'{peaks_folder}selected_{date_time}_{channel}.pkl')
        except :
            print(f'no peaks found for {date_time}, {channel}')
    else : 
        try : 
            peaks = read_dict(f'{peaks_folder}_{date_time}_{channel}.pkl')
        except :
            print(f'no peaks found for {date_time}, {channel}')

    if peaks != [] :
        # We initialize the variables
        frame_index = peaks.frame_index
        amplitude = peaks.amplitude
        power = peaks.power 
    else : 
        frame_index = {'interictal' : [], 'MUA' : []}
        amplitude =  {'interictal' : [], 'MUA' : []}
        power =  {'interictal' : [], 'MUA' : []}

    return frame_index, amplitude, power 

def rewrite_peaks_pkl(date_time, channel, peaks, deleted) : 
    '''
    CAREFUL this deletes the old version of the peaks file 
    This functions deletes some peaks from the peaks object and save this new version 
    Inputs : 
    date_time : date time of the recording
    channel: the channel of interest 
    peaks: the peaks object 
    deleted: list of frames to delete 

    No return, saves the new pickle 
    '''
    for peak_type in peak_types : 
        if len(peaks.frame_index[peak_type]) > 0 : 
            peaks.frame_index[peak_type] = [peaks.frame_index[peak_type][i] for i in range (len(peaks.frame_index[peak_type])) if i not in deleted[peak_type]] # copy the peaks except the ones to delete 
            peaks.power[peak_type] = [peaks.power[peak_type][i] for i in range (len(peaks.power[peak_type])) if i not in deleted[peak_type]] # same for power 
            for data_type in data_types : 
                peaks.amplitude[peak_type][data_type] = [peaks.amplitude[peak_type][data_type][i] for i in range (len(peaks.amplitude[peak_type][data_type])) if i not in deleted[peak_type]] # same for amplitude
    
    # saves the pickle 
    save_dict(peaks, f'{peaks_folder}_{date_time}_{channel}.pkl')


def get_random_peak(peak_type, time_window = freqs, path = None, channel = None) : 
    '''
    This function is used to find a peak in the data 
    # Inputs
    peak_type: peak_type of interest (MUA/IILDs) 
    time_window: the peak will be in the middle of a time window, this specifies the size of this window (in frames)
    path: if we want the peak to be from a specific file, put the path here
    channel: if we want the peak to be from a specific channel, put the channel name here
    # Outputs 
    path: path of the file with the peak 
    channel: channel name 
    frames: [frame start, frame stop], there is a time_window frames between start and stop with the peak in the middle 
    '''

    # get a list of paths and their associated channels by looking at the info file 
    paths, channels_to_keep_by_path = paths_channels_from_info()

    if path is None :
        path = paths[randint(0,len(paths)-1)] # get a random path in paths 

    if channel is None :
        channel = channels_to_keep_by_path[path][randint(0,len(channels_to_keep_by_path[path])-1)] # get a random channel in the channels for this path 

    date_time = date_time_from_path(path)
    frame_index, _, _ = get_peaks_from_pkl(channel, date_time) # Find the frames with peaks 
    frame = frame_index[peak_type][randint(0,len(frame_index)-1)] # get a random frame for the specified peak type 

    return (path, channel, [int(frame-time_window/2), int(frame+time_window/2)]) 

def get_random_peak_serie(peak_type, condition = None, time_window = freqs, path = None, channel = None, return_peaks = False) : 
    '''
    This function is used to find a peak serie in the data 
    # Inputs
    peak_type: peak_type of interest (MUA/IILDs) 
    condition: if we need a specific condition for the path (baseline, drug, washout..)
    time_window: the peak will be in the middle of a time window, this specifies the size of this window (in frames)
    path: if we want the peak to be from a specific file, put the path here
    channel: if we want the peak to be from a specific channel, put the channel name here
    return_peaks: put True if you need the list of peaks between the frames returned 
    # Outputs 
    path: path of the file with the peak 
    channel: channel name 
    frames: [frame start, frame stop], there is a time_window frames between start and stop with the peak in the middle 
    peaks (optional): list of frames with peaks for the peak_type of interest
    '''

    # get a random path (respecting the condition if there is one) if none was given 
    if path is None :
        if condition is not None : 
            paths, _, conditions_by_path = paths_channels_from_info(condition = True)
            paths = [path for path in paths if conditions_by_path[path] == condition]
        else : 
            paths, _ = paths_channels_from_info(condition = False)

        path = paths[randint(0,len(paths)-1)]

    # Get the best channel for this path and peak_type if none was given 
    date_time = date_time_from_path(path)
    if channel is None :
        channel = channels_ordered_from_results(date_time, peak_type)[0]
    
    # Get the frames with peaks 
    frame_index, _, _ = get_peaks_from_pkl(channel, date_time)
    
    # Find frames that are close together in time (at least 2 peaks in half the time window)
    time_diff = np.diff(frame_index[peak_type])
    indices = np.where(time_diff < time_window/2)[0]
    random_index = np.random.choice(indices)

    # frames start is half a time window before the peak chosen and frame stop half time window after 
    frames = [max(int(frame_index[peak_type][random_index]-time_window/2),0),max(int(frame_index[peak_type][random_index]+time_window/2),time_window/2)]

    # If asked, will return the list of frames with peaks within the time window 
    if return_peaks :
        peaks = [f for f in frame_index[peak_type] if f in range (frames[0], frames[1])]
        return (path, channel, frames , peaks)
    else :
        return (path, channel, frames)


## GUI FUNCTIONS

def make_choices(dict_object: dict, text_keys: list, checkbox_keys: list) : 
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


def make_it_interactive(fig: object, axes: object, param: dict) :
    '''
    Function to add sliding bars to zoom on y axis and scroll through x axis; 
    also buttons to move from one peak to another and delete it if needed

    Inputs 
    fig: matplotlib figure to modify
    axes: matplotlib axes to modify 
    param (dict): dictionnaries with the keys:
            'xlim' (list of 2 int): [min_x, max_x]
            'xlabel' (str): label for x axis 
            'ylim' (list of 2 int): [min_y, max_y]
            'ylabel' (str): label for y axis 
            'x_window'(int) : time scale for plot
            'ticks' (list of int): list of points that need to be highlighted on x axis
            'y_zoom' (list of 2 int): min and max zoom on y axis 

    Output
    peaks_to_del (list of int): indexes of peaks to delete 
    '''
    # Adjust figsize to make room for sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Position each slider ([x,y,w,h] x = 0 is left and y = 0 is bottom, width goes right and height goes up)
    ax_slide_x = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    ax_slide_y = fig.add_axes([0.15, 0.25, 0.03, 0.65])

    # Adding ticks if they are defined 
    ax_slide_x.add_artist(ax_slide_x.xaxis)
    ax_slide_x.set_xticks(param['ticks'])
    ax_slide_x.set_xticklabels(["*"]*len(param['ticks'])) # Shown as stars on the plot 
    
    # Create the slider objects based on the defined parameters for label, min and max values. 
    slider_x = wdg.Slider(ax=ax_slide_x, label=param['xlabel'], 
                                            valmin=param['xlim'][0], valmax=param['xlim'][1]- param['x_window'], 
                                            valinit=param['xlim'][0], orientation='horizontal')

    slider_y = wdg.Slider(ax=ax_slide_y, label=param['ylabel'], 
                                            valmin=param['y_zoom'][0], valmax=param['y_zoom'][1], 
                                            valinit=1, orientation='vertical')

    # Create buttons at the right position with labels
    # Previous button on the bottom left 
    ax_button_prev = fig.add_axes([0.25, 0.000001, 0.1, 0.08])
    prev = wdg.Button(ax_button_prev, 'Previous', color= 'grey', hovercolor='blue')

    # Delete button in the middle 
    ax_button_del = fig.add_axes([0.55, 0.000001, 0.1, 0.08])
    delete = wdg.Button(ax_button_del, 'Delete this peak', color= 'grey', hovercolor='red')

    # Next button on the bottom right 
    ax_button_next = fig.add_axes([0.85, 0.000001, 0.1, 0.08])
    next_ev = wdg.Button(ax_button_next, 'Next', color= 'grey', hovercolor='blue')
    

    # The function to be called anytime a slider's value changes
    def update_x(val):
        for ax in axes:
            ax.set_xlim([int(slider_x.val), int(slider_x.val + param['x_window'])]) # scroll through the horizontal axis based on x_window step 
        fig.canvas.draw_idle()

    def update_y(val):
        for s,ax in enumerate(axes) :
            ax.set_ylim([int(param['ylim'][s][0]*(slider_y.val)), int(param['ylim'][s][1]*slider_y.val)]) # Zoom on y axis 
        fig.canvas.draw_idle()

    # Need to create a class to update variables with button presses 
    class IndexPeak:
        # Start at i = -1 so when we click 'next' we go to the next peak (i=0)
        i_peak = -1
        peaks_to_del = []
        colors_ticks = ['black']*len(param['ticks'])
        pause = True

        # Function to go to the next peak 
        def nextone(self, event):
            if self.i_peak < len(param['ticks']) - 1 : # Move to the next peak if we are not at the last one 
                self.i_peak += 1 
            else : # Go back to the first one if we reached the end 
                self.i_peak = 0 
            slider_x.set_val(param['ticks'][self.i_peak] - int((param['x_window'])/2))
            fig.canvas.draw_idle()
            
        # Function to go to the previous peak 
        def previousone(self, event):
            if self.i_peak > 0: # Move to the peak before if we are not at the first one
                self.i_peak -= 1 
            else : # Go to the last one if we reached the beginning
                self.i_peak = len(param['ticks']) - 1 
            slider_x.set_val(param['ticks'][self.i_peak] - int((param['x_window'])/2))
            fig.canvas.draw_idle()

        # Function to delete the peak showed on the screen
        def deletethisone(self, event) : 
            self.peaks_to_del.append(self.i_peak) # We add the index of this peak to a list of peaks to delete 
            self.colors_ticks[self.i_peak] = 'red' # We put a red tick for this one to show that it has been deleted 
            for ticklabel, tickcolor in zip(ax_slide_x.get_xticklabels(), self.colors_ticks):
                ticklabel.set_color(tickcolor)
            if self.i_peak < len(param['ticks']) - 1 : # Move to the next peak if we are not at the last one 
                self.i_peak += 1 
            else : # Go back to the first one if we reached the end 
                self.i_peak = 0 
            slider_x.set_val(param['ticks'][self.i_peak] - int((param['x_window'])/2))
            fig.canvas.draw_idle()

        # Function to return the list of peaks to delete 
        def on_close(self, event) : 
            self.pause = False 


    # Call the function for each update 
    slider_x.on_changed(update_x)
    slider_y.on_changed(update_y)

    callback = IndexPeak()
    
    prev.on_clicked(callback.previousone)
    next_ev.on_clicked(callback.nextone)
    delete.on_clicked(callback.deletethisone)
    fig.canvas.mpl_connect('close_event', callback.on_close)

    while callback.pause :
        plt.pause(2)


    # Show the plot 
    plt.show()
    
    # return the updated figure with sliding bars 
    return callback.peaks_to_del


def choose_files_from_folder() :
    '''
    Ask the user to fill in the path of a folder with h5 files to look at 
    And then shows the list of h5 files in this folder so we can select which we want 

    No input

    Ouputs the list of paths chosen by the user 
    '''
     # Ask the user what is the folder with the recordings 
    gui_dict = {'Path' : {'label' : 'Path of the H5 folder', 'var' : [],'value' : ''}}
    gui_dict = make_choices(gui_dict, text_keys = ['Path'], checkbox_keys = [])
    
    # Save the file path in r format to avoid problems with '\'
    folder = r'{}'.format(gui_dict['Path']['value'])
    files = listdir(folder)
    files = [r'{}'.format(file) for file in files if '.h5' in file]

    # No h5 files in the list 
    if len(files) == 0 : 
        print('Wrong folder? No h5 files found here')

    # For readibility, we show only the date_time of the recording to the user so he/she can chooses 
    for file in files :
        date_time = file[:file.index('Mcs')].replace('T', '_')
        gui_dict[file] = {'label' : date_time, 'var' : [],'value' : 0}
    gui_dict = make_choices(gui_dict, text_keys = [], checkbox_keys = files) # GUI

    # All the checkbox clicked on will have a '1' value so we can save the path selected by the user 
    paths = []
    for file in files :
        if gui_dict[file]['value'] == 1 :
            paths.append(folder + '\\' + file )

    return paths

def choose_files_from_info() :
    '''
    Get the list from file in the info file and the user can click on the ones of interest
    No input
    Ouput the list of paths selected by the user 
    '''

    # Find files from info file 
    paths_info, _ = paths_channels_from_info()
    # Save the file path in r format to avoid problems with '\'
    gui_dict = {}

    # For readibility, we show only the date_time of the recording to the user so he/she can chooses 
    for path in paths_info :
        date_time = date_time_from_path(path)
        gui_dict[path] = {'label' : date_time, 'var' : [],'value' : 0}

    gui_dict = make_choices(gui_dict, text_keys = [], checkbox_keys = paths_info) # GUI 

    # All the checkbox clicked on will have a '1' value so we can save the path selected by the user 
    paths = []
    for path in paths_info :
        if gui_dict[path]['value'] == 1 :
            paths.append(path)

    return paths 
    

# Manipulating channels and their positions 

def listing_channels() :
    '''
    This function prints the name of channels in the 'sparse' and the 'dense' layout (both have 120 channels)
    The sparse layout is a square, columns from A to M and rows from 1 to 10 
    The dense layout is more circular, columns from A to M and rows from 1 to 12 but not all combinations are represented 
    '''
    list_channels = {}
    # List of letters in the alphabet without the I (no I in the channels' names)
    alphabet = list(string.ascii_uppercase)
    i = alphabet.index('I')
    del alphabet[i]
    alphabet = alphabet[:alphabet.index('N')] # We go only to M (last column in both layouts)
    list_channels['sparse'] = [f'{letter}{number}' for letter in alphabet for number in range (1,11)] # sparse is a square
    todel = [f'{l}{i}' for i in range (1,4) for l in ['A', 'M']]
    todel += [f'{l}{i}' for i in range (1,3) for l in ['B', 'L']]
    todel += [f'{l}1' for l in ['C', 'K']]
    todel += [f'{l}{i}' for i in range (10,13) for l in ['A', 'M']]
    todel += [f'{l}{i}' for i in range (11,13) for l in ['B', 'L']]
    todel += [f'{l}12' for l in ['C', 'K']]
    list_channels['dense'] = [f'{letter}{number}' for letter in alphabet for number in range (1,13) if f'{letter}{number}' not in todel] # dense layout goes up to 12 but we delete some combinations 
    print(list_channels)


def channels_around(channels:string) -> string:
    '''
    Input: string of list of channels 
    Output: string of list of the channels given + all the channels in the neighborhood 
    '''

    # List of letters in the alphabet without the I (no I in the channels' names)
    alphabet = list(string.ascii_uppercase)
    i = alphabet.index('I')
    del alphabet[i]

    # We find all the numbers and letters in the channel list 
    numbers = re.findall(r'\d+',channels)
    numbers = [int(number) for number in numbers]
    letters = re.findall(r'[a-zA-Z]',channels)

    # We write a new list starting from one letter and one number before the min channel and with one more number and letter than the max channel
    channels_to_keep = ''
    min_letter = alphabet.index(min(letters))
    max_letter = alphabet.index(max(letters)) 
    for n in range (min(numbers) - 1, max(numbers) + 2) : 
        for letter in range (min_letter - 1, max_letter + 2) :
            channels_to_keep += f'{alphabet[letter]}{n},'

    channels_to_keep = channels_to_keep[:-1]
    return channels_to_keep


def get_channels_positions(date_time: string) : 
    '''
    Inputs: date_time date and time of the file of interest 
        
    Outputs channels list complete
            x coordinates of the channels
            y coordinates of the channels 
    '''

    # open the file with the positions at the right sheet 
    wb = op.load_workbook(positions_file)
    sheet = wb[date_time]

    # Initialize the variables 
    row_0 = 2
    channels = []
    x=[]
    y=[]

    # Get the x,y values from the file 
    for row in range (row_0, sheet.max_row+1): 
        channels.append(sheet.cell(row=row, column=1).value)
        x.append(sheet.cell(row=row, column=2).value)
        y.append(sheet.cell(row=row, column=3).value)

    # save the file 
    wb.save(positions_file)

    return channels, x,y

def except_channels(start_row = start_row) :
    '''
    Change the info file so that the channels written are replaced by all the channels not written here 
    Useful if already analysed the channels there and want to analyze new ones 

    Input
    start_row: row where to start rewriting file 

    No output just directly changes the info file 
    '''

    wb = op.load_workbook(info_file)
    sheet = wb['conditions']

    # Finds all the columns of interest, containing info on the channels to analyse, the layout used, the channels with noise (bruit in french)...
    for col in range (1,sheet.max_column) :
        if sheet.cell(row=1, column=col).value in ['channels','Channels'] :
            channel_col = col 
        if sheet.cell(row=1, column=col).value in ['layout','Layout'] :
            layout_col = col
        if sheet.cell(row=1, column=col).value in ['bruit','Bruit'] :
            noise_col = col

    # For all the rows after the start_row, 

    for row in range (start_row,sheet.max_row+1) : 
        list_channels = []
        for col in [channel_col, noise_col] : 
            channels = sheet.cell(row=row, column= col).value # find the list of channels 
            if type(channels) != type(None) : # if not empty 
                list_channels.extend(channels.replace(' ', '').split(',')) # turns the string to a list of the channels 
        
        # get the layout and all the channels of this layout that were not in the list of channels 
        layout = sheet.cell(row=row, column=layout_col).value
        new_channels = [channel for channel in list_channels_all[layout] if channel not in list_channels]

        # Write the list as a string 
        sheet.cell(row=row, column=channel_col).value = ', '.join(new_channels)

    # Save the file 
    wb.save(info_file)

# Getting info from files 

def paths_channels_from_info(condition = False, start_row = start_row) : 
    '''
    Starting at the start_row defined in the parameter file we get the paths and the list of channels that need to be analysed 
    No inputs
    Output : list of paths of interest and dictionnary with paths as keys and channels of interest for each 
    '''

    # Read the info file at the right sheet 
    wb = op.load_workbook(info_file)
    sheet = wb['conditions']

    # Initialize variables 
    paths = []
    channels_to_keep_by_path = {}
    conditions_by_path = {}

    # Get the paths in the first column and the channels in the second column starting from start_row 
    for row in range (start_row, sheet.max_row + 1) : 
        path = sheet.cell(row=row, column=1).value
        if path is not None:
            paths.append(r'{}'.format(path))
            channels_to_keep_by_path[path] = sheet.cell(row=row, column=2).value.replace(' ', '').split(',')
            if condition :
                conditions_by_path[path] = sheet.cell(row=row, column=3).value

    # Save the file 
    wb.save(info_file)
    
    if condition : 
        return paths, channels_to_keep_by_path, conditions_by_path
    else :
        return paths, channels_to_keep_by_path

def get_image_from_info(date_time: string) : 
    '''
    Input: date_time the date_time of the file of interest 
    Output: the path where the image of the slice recorded at this date_time 
    '''

    # Reads the info file at the right sheet 
    wb = op.load_workbook(info_file)
    sheet = wb['conditions']

    date_time = date_time.replace('_','T')

    # Look for the right date and time and get the number of the slice in the 4th column 
    for row in range (start_row, sheet.max_row + 1) : 
        path_found = sheet.cell(row=row, column=1).value
        if type(path_found) == str and date_time in path_found:
            n = sheet.cell(row=row, column=4).value
            path = path_found

    # Save the file 
    wb.save(info_file)

    folder_img = dirname(path)
    
    for f in format_slices :
        image = r'{}'.format(f'{folder_img}\\slice{n}.{f}')
        path_image = Path(image)
        if path_image.is_file() :
            return (image)
    
    print(f'{folder_img}\\slice{n} not found')

def date_time_from_path(path:string) -> string: 
    '''
    Input: path the path to a h5 file
    Output: date_time as YYYY-MM-DD_HH-MM-SS
    '''
    namefile = basename(normpath(path)) # Get the name of the file from the path 
    date_time = namefile[:namefile.index('Mcs')].replace('T', '_') # Only keep the characters before 'Mcs' and replace the 'T' with '_'

    return date_time

def path_from_datetime(date_time:string) -> string:
    '''
    Inputs: date_time as YYYY-MM-DD_HH-MM-SS 
    Output: path of the h5 file corresponding to date_time given 
    '''
    paths, _ = paths_channels_from_info() # Get a list of all the paths 
    for path in paths :
        date_time = date_time.replace('_','T') # We save the date_time with an underscore but in the paths there is a 'T' instead 
        if date_time in path : # Return the path if it corresponds to the date_time given
            return path

def get_minutes_duration(path) : 
    '''
    From a h5 file, finds the number of frames and convert it to minutes 
    Input: path (str) 
    Output: Number of minutes of this recording
    '''
    raw, _ = raw_signals_from_file(path)

    # We find the length of data stored for the first channel 
    total_frames = len(raw[0])

    # Need to divide the number of frames by the acquisition frequency and 60 to get minutes 
    return total_frames/(60*freqs)

def add_duration_to_infos() :
    '''
    Modifies the info file to add the duration in minutes in the last column based on the number of frames 
    '''

    # open the info file 
    wb = op.load_workbook(info_file)
    sheet = wb['conditions']
    maxcol = sheet.max_column + 1

    # Find the path in the first column of each row 
    for row in range (start_row, sheet.max_row + 1) : 
        path = sheet.cell(row=row, column=1).value
        if path is not None:
            # Get the duration and write it 
            minutes = get_minutes_duration(path)
            sheet.cell(row=row, column=maxcol).value = minutes 

    # Save the file 
    wb.save(info_file)
    
def frames_to_use(date_time) :
    '''
    Reads the info file to find the frames of interest for this file 

    Input : date_time = date time of the file of interest
    Output : l = [start frame, stop frame, list of bad frames]
    # If nothing found, start and stop will be None values and bad frames an empty list 
    '''
    
    # Reads the info file at the right sheet 
    wb = op.load_workbook(info_file)
    sheet = wb['conditions']

    date_time = date_time.replace('_','T')
    
    # Finds the column with the start/stop/bad frames info (they are named like this in the excel file)
    keys = 'start', 'stop', 'bad'
    columns = {}
    for col in range (1,sheet.max_column + 1) :
        for key in keys :
            if sheet.cell(row=1, column=col).value == key :
                columns[key] = col

    # Start with an empty list 
    l = [None,None,[]]

    if columns != {} : 
        # Look for the right date and time 
        for row in range (start_row, sheet.max_row + 1) : 
            path_found = sheet.cell(row=row, column=1).value
            if type(path_found) == str and date_time in path_found:
                for n,key in enumerate(keys) :
                    if key != 'bad' : 
                        k = sheet.cell(row=row, column=columns[key]).value 
                        if type(k) in [int, float] :
                            l[n] = (freqs*k) # converts seconds to frames 
                    
                    else : # For the bad frames, it is not a unique frame but a list of frames sometimes 
                        k = sheet.cell(row=row, column=columns[key]).value
                        if type(k) == int :
                            l[n] = ([freqs*k])
                        elif type(k) == str :
                            k = k.replace(' ','').split(',')     
                            l[n] =[freqs*int(ki) for ki in k]

    return l 

def bad_channels(date_time) : 
    '''
    Reads the info file to find the channels with noise for this file 

    Input : date_time = date time of the file of interest
    Output : list of bad channels 
    
    '''
    wb = op.load_workbook(info_file)
    sheet = wb['conditions']

    date_time = date_time.replace('_','T')

    # Finds the column with noisy channels 
    noise_col = None
    for col in range (1,sheet.max_column) :
        if sheet.cell(row=1, column=col).value in ['bruit','Bruit'] :
            noise_col = col
    
    # If the right column was found, find the row for the file of interest and returns a list of the bad channels 
    if noise_col is not None : 
        for row in range (start_row, sheet.max_row + 1) : 
            path_found = sheet.cell(row=row, column=1).value
            if type(path_found) == str and date_time in path_found:
                bad = sheet.cell(row=row, column=noise_col).value
                bad = bad.replace(' ','').split(',')   # this turns a string list to a list of strings   
    else :
        bad = []

    return bad 


def get_channels_and_z(date_time, peak_criteria, peak_type) : 
    '''
    This functions finds all the pkl file corresponding to the date_time of interest (meaning for all channels) and get the channel name and peak info from this 
    Input: date_time 
    peak_criteria: one of the following ['frequency', 'amplitude', 'power']
    peak_type: interictal/MUA

    Output 
    channels: list of channels analysed 
    z: the value of interest for each channel 
    channels and z are in the same order so easy to know which value corresponds to which channel and vice versa
    '''

    # List of all the files with the right date and time (different channels)
    list_peaks = list_date_time_pkl(date_time)

    channels = []
    z = []

    # For each finds the name of the channel and the information depending on the peak criteria (frequency, median amplitude or median power of the peaks for the channel)
    for file in list_peaks :
        channel = channel_from_pkl(date_time, file)
        channels.append(channel)
        z.append(peak_info(date_time, channel, peak_type, peak_criteria))
    
    return channels, z 

def peak_info(date_time, channel, peak_type, peak_criteria) : 
    '''
    Input: date_time 
    channel: name of the channel
    peak_criteria: one of the following ['frequency', 'amplitude', 'power']
    peak_type: interictal/MUA
    '''

    start, stop, bad = frames_to_use(date_time) # in frames 
    
    start, stop, bad = frames_to_use(date_time) # in frames 

    if start is None :
        start = 0

    if stop is not None and start + after_time + duration > stop :
        print(f'ERROR in duration for {date_time}, start = {(stop-duration)/(60*freqs)} min instead of {(start+after_time)/(60*freqs)}')
        start = stop - duration 
    else :
        start += after_time
        stop = start + duration

    # opens the file with the peak object 
    peaks = read_dict(f'{peaks_folder}_{date_time}_{channel}.pkl')

    frames = peaks.frame_index[peak_type]
    amplitudes_raw = peaks.amplitude[peak_type]['raw']
    power = peaks.power[peak_type]
    
    frequency = 0
    amplitude = 0
    power = 0
    
    if frames != [] and not np.isnan(frames).all(): 
        
        indices = [i for i in range (len(frames)) if frames[i] >= start and frames[i] < stop]
        todel = []
        for j in range (len(bad)) :
            for i in indices :
                if i in range(bad[j]-5*freqs,bad[j]+5*freqs) : 
                    todel.append(i)
        indices = [i for i in indices if i not in todel]
        if len(indices) > 0 : 
            amplitude = np.nanmedian([abs(amplitudes_raw[i]) for i in indices])
            list_powers = []
            for i in indices : 
                try :
                    list_powers.append(10*log10(power[i]*1e-12)) # Converts average power in µV^2 to dbW
                except  :
                    list_powers.append(np.nan)
            power = np.nanmedian(list_powers)
            frequency = len(indices)*freqs/duration
    
                        

    # finds the information of interest based on the criteria defined 
    if peak_criteria == 'frequency' : 
        return frequency
    elif peak_criteria == 'amplitude' : 
        return amplitude
    elif peak_criteria == 'power' : 
        return power
    else : 
        print('peak criteria not found, use amplitude, frequency or power')

    
### Organize our results 

def write_raw_results_file(date_time, peak_type, threshold = None) :
    '''
    Get the results from a recording of interest: the number of channels activated during this recording (superior to the frequency threshold), 
                                                the frequency, amplitude and name of the channel with the max frequency

    Inputs: date_time of the recording
            peak_type of interest 
            the threshold for a channel to be considered active 
    '''

    if threshold is None : 
        threshold = min_freq[peak_type]

    add_sheet(results_file, f'raw_results_{peak_type}')

    list_analyzed = list_date_time_pkl(date_time)

    channels = []
    frequencies = []
    amplitudes = []
    powers = []
    n_channels = 0

    # The headers are the cells from the first row of the excel file 
    for file in list_analyzed :
        channel = channel_from_pkl(date_time, file)
        channels.append(channel)
        peaks = read_dict(peaks_folder + file)
        freq = freqs*len(peaks.frame_index[peak_type])/peaks.total_frames
        frequencies.append(freq)
        amplitudes.append(np.nanmedian(peaks.amplitude[peak_type]['raw']))
        powers.append(np.nanmedian(peaks.power[peak_type]))
        if freq > threshold :
            n_channels += 1 # If its frequency is above the threshold, we count it as active 

    if channels != [] : # If there is at least one channel found 
        max_freq = np.nanmax(frequencies) # We find the max frequency 
        channel_max = channels[frequencies.index(max_freq)] # the channel with this max frequency
        ampli_max = amplitudes[frequencies.index(max_freq)] # its median amplitude 
        median_freq = np.nanmedian(frequencies)
        ampli_median = np.nanmedian(amplitudes)
        ordered_channels = [channels[i] for i in np.argsort(frequencies) if not np.isnan(frequencies[i])]
        ordered_channels.reverse()
        ordered_channels = ', '.join(ordered_channels)
        # We write the results found in a new row of the results file 
        wb = op.load_workbook(results_file)
        
        sheet = wb[f'raw_results_{peak_type}']

        row = sheet.max_row + 1
        sheet.cell(row=row, column=1).value = date_time
        sheet.cell(row=row, column=2).value = n_channels
        sheet.cell(row=row, column=3).value = max_freq
        sheet.cell(row=row, column=4).value = ampli_max
        sheet.cell(row=row, column=5).value = channel_max
        sheet.cell(row=row, column=6).value = ordered_channels
        sheet.cell(row=row, column=7).value = median_freq
        sheet.cell(row=row, column=8).value = ampli_median

        wb.save(results_file)

def experiments_dict() : 
    '''
    This function puts the informations found in the info_file in a dict 
    No input needed, it opens the info_file 
    Output :
        experiments(dict) : For each date time (key of the dict), we can find the list of channels, the condition of the experiment, the slice number..
    
    '''

    # Open the file with infos 
    wb = op.load_workbook(info_file)
    sheet = wb['conditions']
    row = 2
    experiments = {}
    for row in range (start_row, sheet.max_row+1) :
        path = sheet.cell(row=row, column=1).value
        if path is not None : 
            namefile = basename(normpath(path)) # get the file name from the path 
            date_time = namefile[:namefile.index('Mcs')].replace('T', '_') # Date time is stored as YYYY-MM-DDTHH-MM-SS but we change the 'T' to '_' 
            experiments[date_time] = {}
            for column in range (2,sheet.max_column + 1) :
                # Each column stores a different information indicated in the first row, and we get the value for this info in the actual row 
                header = sheet.cell(row=1, column=column).value 
                value =  sheet.cell(row=row, column=column).value
                # if value is None : 
                #     print(f'No {header} found for {date_time}') 
                if type(value) == str :
                    value = value.replace(' ','')
                if (header == 'channels' or header == 'Channels') and type(value) == str:
                    value = value.split(',') # Channels are stored as a string representing a list '[A1,B1,C1,...]' but we want a real list of channels ['A1','B1','C1', ...]
                experiments[date_time][header] = value # Finally we put the value in the dict 
                

    wb.save(info_file) # save the file and returns the dict 
    return experiments


def exp_ID_dict() : 
    '''
    Creates a dictionnary with the infos and results for each exp ID 
    No input
    Ouput: ID_dict, dictionnary with exp_ID as keys
    '''

    # Get the experiments ordered by date_time of experiments
    experiments = experiments_dict()
    
    # Create a dict object ordered by exp_ID 
    ID_dict = {}

    # Scroll through the experiments 
    for date_time in experiments.keys() :
        # Find the exp ID
        exp_ID = experiments[date_time]['ID']
        
        # Create a field for this exp ID if it's new, with all info of interest classed by their condition (baseline/drug/washout)
        if exp_ID not in ID_dict.keys() :
            ID_dict[exp_ID] = {'date_times' : [date_time]}
            for key in experiments[date_time].keys() : 
                ID_dict[exp_ID][key] = {experiments[date_time]['Condition'] : experiments[date_time][key]}
        else : 
            # If the field already exists just add the new info for each condition
            ID_dict[exp_ID]['date_times'].append(date_time)
            for key in experiments[date_time].keys() : 
                ID_dict[exp_ID][key][experiments[date_time]['Condition']] = experiments[date_time][key]

    return ID_dict


def channels_ordered_from_results(date_time, peak_type) :
    '''
    This function returns a list of the channel from the result file 
    The result file is made as the channels are ordered, the first one has the highest frequency etc 
    Input date_time and peak_type
    Ouput list of the ordered channels 
    '''

    # reads the excel file with the sheet for the right peak type 
    wb = op.load_workbook(results_file)
    sheet = wb[f'raw_results_{peak_type}']

    # Finds the right column 
    for col in range (1, sheet.max_column+1) :
        if sheet.cell(row=1, column=col).value == 'channels_ordered' :
            col_chan = col

    # Finds the right row 
    for row in range (2, sheet.max_row+1) :
        date_time_found =  sheet.cell(row=row, column=1).value # Finds the date time of the experiment
        if date_time_found == date_time :
            channels = sheet.cell(row=row, column=col_chan).value
            channels = channels.replace(' ','').split(',') # converts string to list 

    return channels


def y_from_exp_ID(check, peak_type, channels_to_use = 'best_baseline', return_channels = False, excel = False) : 
    '''
    Inputs: check: dict with : {'criteria': name of the criteria, 'category' : where to find this condition in ID_dict, 'condition': the experimental condition to check}
            multiple_channels: True/False depending on if you want to take only the best channels or the n_best channels stored in ID_dict
    
    Outputs: frequencies: [list of frequencies] for each condition for each exp_ID, amplitudes [list of amplitudes (normalized)] for each condition for each exp_ID. Each points is a value for a period defined in the parameters file 
    NB: baseline, drug, washout are grouped together. They each have points only during the timing conditions (see parameters file) 
    NB bis: all this depends on a baseline/drug/washout experiment type 
    todo define check differently I am not convinced 
    '''

    if excel :
        get_peaks = get_peaks_from_excel
    else :
        get_peaks = get_peaks_from_pkl


    ID_dict = exp_ID_dict() # See funtion description
    exp_valid = [exp_ID for exp_ID in ID_dict.keys() if check['condition'] in ID_dict[exp_ID][check['category']].keys() and ID_dict[exp_ID][check['category']][check['condition']] == check['criteria']]

    frequencies = [[] for _ in range (len(exp_valid))] # Creates a list to store the frequencies
    amplitudes = [[] for _ in range (len(exp_valid))] # Creates a list of to store the amplitudes 
    powers =  [[] for _ in range (len(exp_valid))] # Creates a list of to store the amplitudes 

    for e,exp_ID in enumerate(exp_valid) : # For each experiment (each slice is considered independent generally)
    
        print(f'Processing {exp_ID}')

        frequencies[e] = [[] for _ in range (len(ID_dict[exp_ID]['date_times']))]
        amplitudes[e] = [[] for _ in range (len(ID_dict[exp_ID]['date_times']))]
        powers[e] = [[] for _ in range (len(ID_dict[exp_ID]['date_times']))]

        for d, date_time in enumerate(ID_dict[exp_ID]['date_times']) :
            # Find the time 
            
            start, stop, bad = frames_to_use(date_time) # in frames 

            if start is None :
                start = 0

            if stop is not None and start + after_time + duration > stop :
                print(f'ERROR in duration for {date_time}, start = {(stop-duration)/(60*freqs)} min instead of {(start+after_time)/(60*freqs)}')
                start = stop - duration 
            else :
                start += after_time
                stop = start + duration

            if date_time in ['2023-01-12_16-01-55', '2023-01-12_16-01-56', '2023-01-12_16-01-57'] : 
                date_time = '2023-01-12_16-01-57'

            if channels_to_use == 'best_baseline' and d == 0 : 
                channels = channels_ordered_from_results(date_time, peak_type)[:10]
                
            if channels_to_use == 'all' : 
                channels = channels_ordered_from_results(date_time, peak_type)
            
            if type(channels_to_use) == list : 
                channels = channels_to_use

            freq = [0 for _ in range (len(channels))]
            amp = [np.nan for _ in range (len(channels))]
            puiss = [np.nan for _ in range (len(channels))]

            for c,channel in enumerate(channels) :
                
                frames, amplitudes_c, power = get_peaks(channel, date_time, selected_peaks = False)

                
                frames = frames[peak_type]
                amplitudes_raw = amplitudes_c[peak_type]['raw']
                power = power[peak_type]
                

                if frames != [] and not np.isnan(frames).all(): 
                    
                    indices = [i for i in range (len(frames)) if frames[i] >= start and frames[i] < stop]
                    todel = []
                    for j in range (len(bad)) :
                        for i in indices :
                            if i in range(bad[j]-5*freqs,bad[j]+5*freqs) : 
                                todel.append(i)
                    indices = [i for i in indices if i not in todel]
                    if len(indices) > 0 : 
                        amp[c] = np.nanmedian([abs(amplitudes_raw[i])*factor_amp for i in indices]) # converts weird unit to µV (based on conversion factor from infochannel in the h5 file, change this!!!)
                        list_powers = []
                        for i in indices : 
                            try :
                                list_powers.append(10*log10(power[i]*1e-12)) # Converts average power in µV^2 to dbW
                            except  :
                                list_powers.append(np.nan)
                        puiss[c] = np.nanmedian(list_powers)
                        freq[c] = len(indices)*freqs/duration
                        

            if channels_to_use == 'best_baseline' and d == 0  : 
                best_channels_baseline = list(np.argsort(freq))
                best_channels_baseline.reverse()
                best_channels_baseline = [chan for chan in best_channels_baseline if not np.isnan(freq[chan])][:N_best]
                channels = [channels[chan] for chan in best_channels_baseline]
                freq = [freq[chan] for chan in best_channels_baseline]
                amp = [amp[chan] for chan in best_channels_baseline]
                puiss = [puiss[chan] for chan in best_channels_baseline]
            

            frequencies[e][d] = np.nanmedian(freq)
            amplitudes[e][d] = np.nanmedian(amp)
            powers[e][d] = np.nanmedian(puiss)

    if return_channels :
        return frequencies, amplitudes, powers, channels
    else :
        return frequencies, amplitudes, powers 
    
def surf_from_exp_ID(check, peak_type, excel = False) : 
    '''
    Inputs: check: dict with : {'criteria': name of the criteria, 'category' : where to find this condition in ID_dict, 'condition': the experimental condition to check}
            multiple_channels: True/False depending on if you want to take only the best channels or the n_best channels stored in ID_dict
    
    Outputs: surfaces: [list of surfaces] for each condition for each exp_ID.

    NB: all this depends on a baseline/drug/washout experiment type 

    TODO define check differently I am not convinced 
    '''

    if excel :
        get_peaks = get_peaks_from_excel
    else :
        get_peaks = get_peaks_from_pkl


    ID_dict = exp_ID_dict() # See funtion description
    exp_valid = [exp_ID for exp_ID in ID_dict.keys() if check['condition'] in ID_dict[exp_ID][check['category']].keys() and ID_dict[exp_ID][check['category']][check['condition']] == check['criteria']]

    surfaces = [[] for _ in range (len(exp_valid))] 

    for e,exp_ID in enumerate(exp_valid) : # For each experiment (each slice is considered independent generally)
    
        print(f'Processing {exp_ID}')

        surfaces[e] = [0 for _ in range (len(ID_dict[exp_ID]['date_times']))]

        for d, date_time in enumerate(ID_dict[exp_ID]['date_times']) :
            # Find the time 
            
            start, stop, bad = frames_to_use(date_time) # in frames 

            if start is None :
                start = 0

            if stop is not None and start + after_time + duration > stop :
                print(f'ERROR in duration for {date_time}, start = {(stop-duration)/(60*freqs)} min instead of {(start+after_time)/(60*freqs)}')
                start = stop - duration 
            else :
                start += after_time
                stop = start + duration

            if date_time in ['2023-01-12_16-01-55', '2023-01-12_16-01-56', '2023-01-12_16-01-57'] : 
                date_time = '2023-01-12_16-01-57'

            channels = list_channels_all['sparse']

            for channel in channels : 
                
                frames, _, _ = get_peaks(channel, date_time, selected_peaks = False)

                
                frames = frames[peak_type]
                freq = 0 
                if frames != [] and not np.isnan(frames).all(): 
                    
                    indices = [i for i in range (len(frames)) if frames[i] >= start and frames[i] < stop]
                    todel = []
                    for j in range (len(bad)) :
                        for i in indices :
                            if i in range(bad[j]-5*freqs,bad[j]+5*freqs) : 
                                todel.append(i)
                    indices = [i for i in indices if i not in todel]
                    freq = len(indices)*freqs/duration
                
                if freq > min_freq[peak_type] : 
                    surfaces[e][d] += 1 
            
            surfaces[e][d] = surfaces[e][d]*1.5

    return surfaces 

def get_peaks_from_excel(channel, date_time, find_informations = False, selected_peaks = False) : 
    '''
    Read excel file to find peaks (old)
    '''
    # We initialize the variables
    frame_index = {}
    amplitude = {}
    power = {}

    if find_informations : 
        infos = {}

    for peak_type in peak_types :
        frame_index[peak_type] = []
        amplitude[peak_type] = {}
        power[peak_type] = []
        for data_type in data_types : 
            amplitude[peak_type][data_type] = []

    take = False 

    if selected_peaks :
        sheet = pd.read_excel(selected_peaks_file, sheet_name=date_time)
    else :
        sheet = pd.read_excel(peaks_file, sheet_name=date_time)

    # The headers are the cells from the first row of the excel file 
    for header in sheet.keys() :
        if 'channel'  in header :
            take = False
            if sheet[header][0] == channel :
                index_peak_type = list(sheet[header]).index('peak_type')
                peak_type = sheet[header][index_peak_type+1]
                if peak_type in peak_types :
                    take = True

                    if find_informations : 
                        infos[peak_type] = list(sheet[header])

        if 'frame' in header and take : #need to take all the frames where there are peaks (column with 'frame' as header)
                frame_index[peak_type] = list(sheet[header])

        for data_type in data_types :
            if data_type in header and take : #need to take the amplitudes for each data type 
                amplitude[peak_type][data_type] = list(sheet[header])
    

    if find_informations : 
        return frame_index, amplitude, power, infos
    else :
        return frame_index, amplitude, power

def statistics(y, show_ns = False, print_stats = False) :
    '''
    From a list of results y (with each element of y are the same size), returns statistics 
    Only works on paired test for now, need to update (todo)
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


## PLOTTING 

   
        
def final_plotting(ys, conditions, check, peak_type, names = ['Frequency (Hz)', 'Amplitude (µV)', 'Power (dbW)'], normalize_to_baseline = False, limy = None, time_limit = True, saving = False, timeplot = True, broken = None, no_washout = False, subplots = False, fig = None, axes = None, yticks = None, xticks = None) : 
    '''
    This function is used to make all the boxplots from ys
    ys : must be a list of subplots each containing a trial each containing a condition (baseline, drug, washout)
    check: dict with {'criteria': name of the criteria, 'category' : where to find this condition in ID_dict, 'condition': the experimental condition to check}
            multiple_channels: True/False depending on if you want to take only the best channels or the n_best channels stored in ID_dict

    todo: commment more and adapt for time_limit and plot in time do not work anymore, need to put them in other function see pipeline report task3
    '''

    print(f'{peak_type} - {check["criteria"]} - {len(ys[0])} trials')


    if subplots and axes is None :
        fig, axes = start_fig(nrows = 1, ncols = len(names))

    for plot in range (len(ys)) : 
        if time_limit : 
            y = time_limiting(ys[plot])
            if timeplot :
                plot_in_time(y, conditions, limy = limy[plot], only_median = True, put_rectangles = True)

        else :
            y = ys[plot]
            if timeplot :
                plot_in_time(y, conditions, limy = limy[plot], only_median = False, put_rectangles = True)

        box = [[] for _ in range (len(y))]
        for e,exp in enumerate(y) : 
            box[e] = [[] for _ in range (len(exp))]
            for c,condition in enumerate(exp) :
                box[e][c] = np.nanmedian(condition)

        box = [[box[i][j] for i in range (len(box))] for j in range (len(box[0]))]

        if no_washout :
            box = box[:2]
            conditions = conditions[:2]

        stars = statistics(box, show_ns = False, print_stats = True)

        if normalize_to_baseline :
            ref = np.nanmedian(box[0])
            if np.isnan(ref) or ref == 0 :
                ref = 1
                print('!!!NO NORMALIZATION!!!!')
            box = [[boxi[i]/ref for i in range (len(boxi))] for boxi in box]
            print(box)

        if subplots :
            axes[plot] = make_boxplot(box, conditions, draw_line=True, namefig=f'{check["criteria"]}_{peak_type}_{names[plot]}', stars = stars, limy = limy[plot], saving = saving, broken = broken[plot], fig = fig, ax = axes[plot], show = False, yticks = yticks[plot], ylabel = names[plot], xticks = xticks[plot])

        else :
            make_boxplot(box, conditions, draw_line=True, namefig=f'{check["criteria"]}_{peak_type}_{names[plot]}', stars = stars, limy = limy[plot], saving = saving, broken = broken[plot], yticks = yticks[plot], ylabel = names[plot], xticks = xticks[plot])

    if subplots :
        return axes 
    else : 
        plt.show()



# Plot signal / peaks 

def select_peaks(path: string, channel:string , peak_types: list, x_window:int = 500, data = None, peaks = None) -> list: 
    '''
    This function creates a plot with the raw signal, the filtered signal and the peaks found 

    Input: 
    path of the recording
    channel of interest 
    x_window: the zoom we want when we look at the peaks (default is 500 so we see 500 ms and we can scroll through the recording) 

    Output:
    deleted: indices of the deleted peaks  

    '''

    if data is None : # Get the raw data
        raw, channels = raw_signals_from_file(path)
        channel_index = channels.index(channel)

        # Take only the signal from the channel chosen
        raw_channel = raw[channel_index]

        # Call the class preprocessing to filter, (denoise) and normalize the signal 
        data = preprocessing(raw_channel, peak_types)
    
    # We put the signals of interest in the signals variable for each data type and peak_types 
    signals = {}
    for data_type in data_types : 
        # Get the filtered and normalised signals for each peak type
        if data_type != 'raw':
                signals[data_type] = data.signal[data_type]
        else : # For the raw data we have the same signal for all peak types 
            signals[data_type] = {}
            for peak_type in peak_types:
                signals[data_type][peak_type] = data.raw

    # We get the data with the peaks information from the excel file 
    date_time = date_time_from_path(path)
    print(f'{date_time}_{channel} processed, getting the peaks')

    if peaks is None :
        peaks = read_dict(f'{peaks_folder}_{date_time}_{channel}.pkl')

    frame_index = peaks.frame_index
    amplitude = peaks.amplitude

    deleted = {}
    

    # For each peak type

    for peak_type in peak_types:
        frame_index[peak_type] = [int(frame) for frame in frame_index[peak_type]  if not isnan(frame)]

        if len(frame_index[peak_type]) > 0 : 

            for data_type in data_types :
                amplitude[peak_type][data_type] = [amp for amp in amplitude[peak_type][data_type] if not isnan(amp)]
                if len(frame_index[peak_type]) != len(amplitude[peak_type][data_type]) :
                    print('ERROR in peak file with number of peaks')
                

            print(f'Plotting the signal with {peak_type} peaks...')
            # Create the figure 
            fig, ax = start_fig(nrows=len(data_types))

            # Plot parameters 
            signal_ref = signals[data_type][peak_type]
            x_ms = [i/10 for i in range (len(signal_ref))]
            param_plot = {
                'xlim' : [x_ms[0], x_ms[-1]],
                'xlabel' : 'Time',
                'ylim' : [[int(min(signals[data_type][peak_type])), int(max(signals[data_type][peak_type]))] for data_type in signals.keys()],
                'ylabel' : 'Amplitude',
                'x_window': x_window,
                'ticks' : [],
                'y_zoom': [0.25,5]   
                }

            # We add the list of peaks to the plot parameters
            if len(frame_index[peak_type]) > 0 :   
                ms_index = [x_ms[i] for i in frame_index[peak_type]]
                param_plot['ticks'] = list(ms_index)

            for s,data_type in enumerate(signals.keys()) :
                ax[s].set_title(f'{data_type} signal')
                # Plot the signal with the right color for each peak type 
                ax[s].plot(x_ms, signals[data_type][peak_type], color = param[peak_type]['color_plot'], alpha = alpha_plots)

                # If peaks were found, show them as stars with the right color for each peak type 
                if len(frame_index[peak_type]) > 0 :   
                    ax[s].scatter(ms_index, np.array(amplitude[peak_type][data_type]), marker = '*', color = param[peak_type]['color_peak'])
                
                if data_type == 'normalized' : 
                    # Show an horizontal line for threshold with the right color for each peak type 
                    ax[s].plot([data.threshold[peak_type] for _ in range(len(signals[data_type][peak_type]))], color = param[peak_type]['color_peak'])
                    
                # Definition of the x and y axes limits 
                ax[s].set_xlim(param_plot['xlim'][0], param_plot['xlim'][0]+param_plot['x_window'])
                ax[s].set_ylim(param_plot['ylim'][s])

            # Create sliding bars to go through different time points, zoom in and out in amplitude and time 
            n_peaks = len(frame_index[peak_type])
            deleted[peak_type] = make_it_interactive(fig, ax, param_plot)
        
            # Print number of peaks deleted
            n_del = len(deleted[peak_type])
            print(f'{n_del} peaks out of {n_peaks} total peaks have been deleted') 

    return peaks, deleted 

def plot_frames_data_type(path: string, channel:string , peak_type: string, frames: list, raster_plot = False, frame_index = None, draw_line = False, color = None, saving= False, format_fig = format_fig, xbar = 0, ybar = 0): 
    '''
    This function plot the data for the specified path and channel with raw data, filtered, normalized and a line indicating the threshold if needed, possible to add a raster plot 
    todo comment more 
    '''

    if color is None : 
        color = param[peak_type]['color_plot']
    
    raw, channels = raw_signals_from_file(path)
    channel_index = channels.index(channel)

    # Take only the signal from the channel chosen
    raw_channel = raw[channel_index][frames[0]:frames[1]]

    # Call the class preprocessing to filter, (denoise) and normalize the signal 
    data = preprocessing(raw_channel, [peak_type])
    
    # We put the signals of interest in the signals variable for each data type and peak_types 
    if raster_plot : 
        if frame_index is None : 
            frame_index, _ = signal_func.find_peaks(data.signal['normalized'][peak_type], height = data.threshold[peak_type], distance = param[peak_type]['inter_event'], width = param[peak_type]['peak_duration'])
        else :
            frame_index = [f-frames[0] for f in frame_index if f in range (frames[0], frames[1])]

    # Create the figure 
    if raster_plot :
        height_ratios = [3 for _ in range (len(data_types)+2)]
        height_ratios[-2] = 1
        fig, ax = start_fig(figsize = (3*cm2inch(fig_len),3*cm2inch(fig_len)), nrows=len(data_types)+2, height_ratios=height_ratios)
    else  : 
        fig, ax = start_fig(figsize = (3*cm2inch(fig_len),2.5*cm2inch(fig_len)), nrows=len(data_types)+1)
    

    
    for s,data_type in enumerate(data_types) : 
        # Get the filtered and normalised signals for each peak type
        if data_type != 'raw':
                signal = data.signal[data_type][peak_type]
        else : # For the raw data we have the same signal for all peak types 
           signal = data.raw
           signal = [raw*factor_amp for raw in signal]
           xlim  = [0, len(signal)]

        # Plot the signal with the right color for each peak type 
        ax[s].plot(signal, color = color)
        ax[s].set_xlim(xlim)
        ax[s] = set_ax_parameters(ax[s], just_plot=True)
        
        if data_type == 'normalized' and draw_line == True : 
            ax[s].plot(xlim, [data.threshold[peak_type],data.threshold[peak_type]], color = 'grey')
    
    if raster_plot : 
        ax[s+1].set_xlim(xlim)
        for frame in frame_index : 
            ax[s+1].plot([frame,frame], [0,1], color = 'black', lw = 2)
        ax[s+1] = set_ax_parameters(ax[s+1], just_plot=True)

    ylim = ax[0].get_ylim()
    ax[-1].plot([xlim[0], xlim[0]+xbar], [ylim[0], ylim[0]], color = 'black')
    ax[-1].plot([xlim[0]+xbar, xlim[0]+xbar], [ylim[0], ylim[0]+ybar], color = 'black')
    ax[-1].set_xlim(xlim)
    ax[-1].set_ylim(ylim)
    ax[-1] = set_ax_parameters(ax[-1], just_plot=True)

    if saving : 
        date_time = date_time_from_path(path)
        plt.savefig(f'{folder_fig}\\{date_time}_data_types_{peak_type}.{format_fig}', format = format_fig, transparent = True)

    plt.show()


def plot_time_spectrogram(path, channel, frames, limy = None, saving = False, format_fig = format_fig, raster_plot = False, frame_index = None, peak_type = 'interictal', show_xticks = False) :
    '''
    This function plot the raw signal between the frames of interest and adds a spectral plot
    todo: update this function so its a part of other data plots (make all of them into a class, where I can do different plots based on the path, channel and frames)
    '''
    raw, channels = raw_signals_from_file(path)
    raw_channel = raw[channels.index(channel)][frames[0]:frames[1]]
    window = signal.windows.kaiser(256, beta=5)
    nperseg = 256
    noverlap = 128
    nfft = 512
    f, t, Sxx = signal.spectrogram(raw_channel, fs = freqs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    frame_index, _, _ = get_peaks_from_pkl('G9', '2022-11-22_23-00-59')
    frame_index = [f for f in frame_index['MUA'] if f in range (frames[0], frames[1])]
    if raster_plot :
        fig, axes = start_fig(nrows = 3, ncols = 1, height_ratios = [4,4,1])
        if frame_index is None : 
            date_time = date_time_from_path(path)
            frame_index, _, _ = get_peaks_from_pkl(channel, date_time)
            frame_index = [f-frames[0] for f in frame_index[peak_type] if f in range (frames[0], frames[1])]
        else :
            frame_index = [f-frames[0] for f in frame_index if f in range (frames[0], frames[1])]
    else  : 
            fig, axes = start_fig(nrows = 2, ncols = 1)

    image = axes[0].pcolormesh(t, f, Sxx, shading='gouraud', cmap= 'inferno')
    axes[1].plot(raw_channel, color = 'black', lw = 0.5)
    fig.colorbar(image, ax=axes[0], location = 'top', fraction=0.6, pad=0)
    axes[1].set_xlim([0,len(raw_channel)])
    if limy is not None :
        axes[0].set_ylim(limy)
    fig.tight_layout(pad=0)


    if raster_plot : 
        xlim = axes[1].get_xlim()
        axes[2].set_xlim(xlim)
        print(len(frame_index))
        for frame in frame_index : 
            axes[2].plot([frame,frame], [0,1], color = 'black', lw = 2)

    if show_xticks :
        l = list(range(0,len(raw_channel),5*freqs))
        xticks = [int(i/freqs) for i in l]
    else  :
        xticks = None

    for i in range (len(axes)) :
        axes[i] = set_ax_parameters(axes[i], just_plot=True, show_xticks=show_xticks, xticks = xticks )

    
        # axes[1].set_xticks(l,l_s)

    if saving : 
        date_time = date_time_from_path(path)
        plt.savefig(f'{folder_fig}\\spectogram_{date_time}_{channel}_{frames[0]}_{frames[1]}.{format_fig}', format = format_fig, transparent = True, peak_type ='interictal')

    plt.show()

def plot_raw_signal(path, channel, frames, limy = None, raster_plot = False, saving = False, format_fig = format_fig, peak_types = ['interictal','MUA'], xbar = None, ybar = None) : 
    '''
    Basic function to plot the raw signal depending on the path of the file of interest, the channel to plot and the frames window [firstframe,lastframe]
    Input
    path : str 
    channel : str
    frames : [frame start, frame stop]
    limy : limits for the y axis [y start, y stop]
    '''
    raw, channels = raw_signals_from_file(path)
    channel_index = channels.index(channel)
    raw_channel = raw[channel_index][frames[0]:frames[1]]
    if raster_plot :
        if len(peak_types) == 1 :
            height_ratios = [3,1,3]
        else :
            height_ratios = [3,1,1,3]
        fig, axes = start_fig(nrows = 2+len(peak_types), height_ratios = height_ratios, figsize = (3*(cm2inch(fig_len)), 4 * (cm2inch(fig_len))))
    else : 
        fig, axes = start_fig(nrows = 2)

    fig.tight_layout(pad=0)
    
    raw_channel = [raw*factor_amp for raw in raw_channel]
    axes[0].plot(raw_channel, color = 'black')
    if limy is not None :
        axes[0].set_ylim(limy)
    axes[0] = set_ax_parameters(axes[0], just_plot=True)
    xlim = axes[0].get_xlim()

    if raster_plot :
        date_time = date_time_from_path(path)
        frame_ind, _, _ = get_peaks_from_pkl(channel, date_time)
        for p,peak_type in enumerate(peak_types) : 
            frame_index = [f-frames[0] for f in frame_ind[peak_type] if f in range (frames[0], frames[1])]
            axes[1+p].set_xlim(xlim)
            for frame in frame_index : 
                axes[1+p].plot([frame,frame], [0,1], color = 'black', lw = 2)
            axes[1+p] = set_ax_parameters(axes[1+p], just_plot=True)
            axes[1+p].set_ylabel(peak_type)

    if xbar is None :
        xbar = (xlim[1]-xlim[0])/10
        print(f'Ref in x: {xbar}')
    if ybar is None :
        ylim = axes[0].get_ylim()
        ybar = (ylim[1]-ylim[0])/10
        print(f'Ref in y: {ybar}')

    axes[-1].plot([xlim[0], xlim[0]+xbar], [0,0], color = 'black')
    axes[-1].set_xlim(xlim)
    axes[-1].plot([xlim[0]+xbar, xlim[0]+xbar], [0,ybar], color = 'black')
    axes[-1].set_ylim(axes[0].get_ylim())
    axes[-1] = set_ax_parameters(axes[-1], just_plot=True)

    if saving : 
        date_time = date_time_from_path(path)
        plt.savefig(f'{folder_fig}\\raw_{date_time}_{channel}.{format_fig}', format = format_fig, transparent = True)
        #plt.savefig(f'{folder_fig}\\raw_axes.{format_fig}', format = format_fig, transparent = True)
    plt.show()


def subplot_all_channels(path, frames = [0,freqs], limy = None, layout = 'sparse', letter_lim = None, number_lim = None, saving = False, namefig = 'all_channels') :
    '''
    This function shows the raw data for all the channels as a big matrix 
    '''
    date_time = date_time_from_path(path)
    try :
        raw_channels = f'{folder_fig}\\{date_time}_raw_channels.pkl'
    except :
        raw_channels = raw_data_all_channels(path, frames, to_return = True)

    alphabet = list(string.ascii_uppercase)
    i = alphabet.index('I')
    del alphabet[i]
    channels = ', '.join(list_channels_all[layout])
    
    if letter_lim is None : 
        letters = re.findall(r'[a-zA-Z]',channels)
        min_letter = alphabet.index(min(letters))
        max_letter = alphabet.index(max(letters))
    else : 
        min_letter = alphabet.index(letter_lim[0])
        max_letter = alphabet.index(letter_lim[1])

    rangeletter = max_letter - min_letter + 1

    if number_lim is None : 
        numbers = re.findall(r'\d+',channels)
        numbers = [int(number) for number in numbers]
    else : 
        numbers = number_lim
    
    rangenumber = max(numbers) - min(numbers) + 1 

    fig, axes = start_fig(nrows = rangenumber, ncols = rangeletter)
    
    fig.tight_layout()

    for n in range (min(numbers), max(numbers) + 1) : 
        for letter in range (min_letter, max_letter + 1) :
            raw_channel = raw_channels[n][alphabet[letter]]
            axes[n-min(numbers)][letter-min_letter].plot(raw_channel, color = 'black')
            if limy is not None :
                axes[n-min(numbers)][letter-min_letter].set_ylim(limy)
            axes[n-min(numbers)][letter-min_letter].set_yticks([],[])
            axes[n-min(numbers)][letter-min_letter].set_xticks([],[])
        print(f'{(n+1)*100/rangenumber}% done', end = '\r')

    if saving :
        plt.savefig(namefig, format = format_fig)
    plt.show()


# Quantitative plots 


def show_colormap_1Z(date_time: string, peak_criteria: string, peak_type: string, saving:bool = False, format_save: str = format_fig, limy = None, yticks = None, show_channels = False) : 
    '''
    date_time date and time of the recording of interest 
    peak_criteria: one of the following ['frequency_interictal', 'freq_intra', 'freq_inter','duration']

    No output 
    '''


    # Find the path of the image from the date_time of the recording 
    image = get_image_from_info(date_time)
    
    #  Get the positions of the channels 
    channels, x,y = get_channels_positions(date_time)

    # get the noisy channels 
    bad = bad_channels(date_time)

    # Find out the letters and numbers present in the channels names 
    alphabet = list(string.ascii_uppercase)
    letters = [letter for letter in alphabet if letter in [channel[0] for channel in channels]]
    numbers = range(1,max([int(channel[1:]) for channel in channels])+1)


    # Initialize the variables 
    X = [[] for _ in range (len(letters))]
    Y = [[] for _ in range (len(numbers))]# We create a grid with the different channel numbers as rows (each row has a different y value in Y)
    Z1 = [0]*len(numbers) # Each channel in the grid has a z value based on the peaks frequency or the duration of the bursts recorded in this channel... 
    
    not_analyzed = []

    

    channels_analysed, z= get_channels_and_z(date_time, peak_criteria, peak_type)
    
    if limy is None : 
        min_z1 = np.nanmin(z)
        max_z1 = np.nanmax(z)
    else : 
        min_z1, max_z1 = limy 


    for number in numbers : 
        Z1[number-1] = [0]*len(letters) 

        
        for l,letter in enumerate(letters): 
        
            channel = f'{letter}{number}' 
            if channel in channels: 
                ind = channels.index(channel) # We find the index of the channel of interest in the original channel list (not ordered)

                X[l].append(x[ind])
                Y[number-1].append(y[ind])

                if channel in channels_analysed and channel not in bad : 
                    ind_analysed = channels_analysed.index(channel)
                    Z1[number-1][l] = z[ind_analysed] # We keep the z value of this channel in the right place 

                else :
                    not_analyzed.append(channel)




    Y= np.nanmedian(Y, axis = 1) # mean y coordinate through all letters for this number 
    X = np.nanmedian(X, axis = 1) # mean x coordinate through all numbers for each letter


    print(f'Channels {not_analyzed} were not analysed')

    # Open the image 
    I = np.asarray(Image.open(image))
    
    # We plot the image and put a transparent (based on alpha value) colormap above it depending on the z values
    _, ax = start_fig(figsize = (3 * (cm2inch(fig_len)), 1.5 * (cm2inch(fig_len))))
    
    
    
    xlim = 0,I.shape[0]
    ylim = 0,I.shape[1]


    Yg, Xg, Z1 = expand_matrix(list(Y), ylim, list(X), xlim, Z1) # put values everywhere for homogeneity (NB: here Z is defined with (Y,X) not (X,Y) be careful)


    if show_channels : 
        for number in numbers: # For each channel number 
            for l,letter in enumerate(letters) : 
                ax.text(X[l], Y[number-1], f'{letter}{number}', color = 'black')
    
    ax.imshow(I)
    
    ax = set_ax_parameters(ax, just_plot=True)

    # # Create the grid based on the computed X,Y values 
    # Xg,Yg = np.meshgrid(Xg,Yg)

    if peak_type == 'interictal' : 
        mapcolor1 = clm.Reds
        mapcolor1.set_under(color='white', alpha = 1)
        mapcolor1 = add_linear_transparency(mapcolor1)

    else :
        print(peak_type)
        mapcolor1 = clm.Blues
        mapcolor1.set_under(color='white', alpha = 1)
        mapcolor1 = add_linear_transparency(mapcolor1)


    for_int = 10000

    step1 = int((for_int*max_z1 - for_int*min_z1)/steps_color)
    levels1 = np.array([i/for_int for i in range (int(for_int*min_z1), int(for_int*max_z1), step1)])
    cb1 = ax.contourf(Xg, Yg, Z1, cmap= mapcolor1, levels = levels1, antialiased=True, vmin = min_z1, vmax =max_z1,  algorithm = 'serial')
    
        
    # Show everything

    # We add a reference colorbar
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0)
    if yticks is None :
        steptick1 = (int(for_int*max_z1) - int(for_int*min_z1))/5
        colorticks1= [round(i/for_int,3) for i in range (int(for_int*min_z1), int(for_int*max_z1)+int(steptick1), int(steptick1))]
    else :
        colorticks1 = yticks
    cbar1 = plt.colorbar(cb1, cax=cax1, ticks = colorticks1)
    cbar1.ax.tick_params(labelsize= fontsize)
    cbar1.set_label('IILD frequency (Hz)', rotation=270, fontsize = 13, labelpad = 20, font = font)
    cbar1.ax.set_yticklabels(colorticks1)

    plt.tight_layout()

    if saving :
        plt.savefig(f'{folder_fig}\\{date_time}_{peak_criteria}.{format_save}', format = format_save, transparent = True)

    plt.show()

def show_colormap_2Z(date_time: string, peak_criteria: string, peak_types, saving:bool = False, format_save: str = format_fig, show_channels = False, ylims = [None, None], yticks = [None, None]) : 
    '''
    date_time date and time of the recording of interest 
    peak_criteria: one of the following ['frequency_interictal', 'freq_intra', 'freq_inter','duration']

    No output 
    todo make 1Z and 2Z as part of 1 function that can show up to 4 colors?
    '''


    # Find the path of the image from the date_time of the recording 
    image = get_image_from_info(date_time)
    
    #  Get the positions of the channels 
    channels, x,y = get_channels_positions(date_time)

    # get the noisy channels 
    bad = bad_channels(date_time)

    # Find out the letters and numbers present in the channels names 
    alphabet = list(string.ascii_uppercase)
    letters = [letter for letter in alphabet if letter in [channel[0] for channel in channels]]
    numbers = range(1,max([int(channel[1:]) for channel in channels])+1)


    # Initialize the variables 
    X = [[] for _ in range (len(letters))]
    Y = [[] for _ in range (len(numbers))]# We create a grid with the different channel numbers as rows (each row has a different y value in Y)
    Z1 = [0]*len(numbers) # Each channel in the grid has a z value based on the peaks frequency or the duration of the bursts recorded in this channel... 
    Z2 = [0]*len(numbers) 
    
    not_analyzed = []

    
    channels_analysed = [0]*len(peak_types)
    z = [0]*len(peak_types)

    # Based on the decided criteria, get the right value for each channel 
    for p,peak_type in enumerate(peak_types) : 
        channels_analysed[p], z[p] = get_channels_and_z(date_time, peak_criteria, peak_type)
    
    if ylims[0] is None : 
        min_z1 = np.nanmin(z[0])
        max_z1 = np.nanmax(z[0])
    else : 
        min_z1, max_z1 = ylims[0]

    if ylims[1] is None : 
        min_z2 = np.nanmin(z[1])
        max_z2 = np.nanmax(z[1])
    else : 
        min_z2, max_z2 = ylims[1]

    for number in numbers : 
        Z1[number-1] = [0]*len(letters) 
        Z2[number-1] = [0]*len(letters)
        
        
        for l,letter in enumerate(letters): 
        
            channel = f'{letter}{number}' 
            if channel in channels: 
                ind = channels.index(channel) # We find the index of the channel of interest in the original channel list (not ordered)

                X[l].append(x[ind])
                Y[number-1].append(y[ind])

                if channel in channels_analysed[0] and channel in channels_analysed[1] and channel not in bad : 
                    ind_analysed = [channels_analysed[0].index(channel),channels_analysed[1].index(channel)]
                    Z1[number-1][l] = z[0][ind_analysed[0]] # We keep the z value of this channel in the right place 
                    Z2[number-1][l] = z[1][ind_analysed[1]]
                    if z[0][ind_analysed[0]] > min_freq[peak_type] or z[1][ind_analysed[1]] > min_freq[peak_type] :
                        print(f'{channel} : ({x[ind]}, {y[ind]}), z = {z[0][ind_analysed[0]],z[1][ind_analysed[1]]}')
                else :
                    not_analyzed.append(channel)




    Y= np.nanmedian(Y, axis = 1) # mean y coordinate through all letters for this number 
    X = np.nanmedian(X, axis = 1) # mean x coordinate through all numbers for each letter


    print(f'Channels {not_analyzed} were not analysed')

    # Open the image 
    I = np.asarray(Image.open(image))
    
    # We plot the image and put a transparent (based on alpha value) colormap above it depending on the z values
    _, ax = start_fig(figsize = (3 * (cm2inch(fig_len)), 2 * (cm2inch(fig_len))))
    
    
    
    xlim = 0,I.shape[0]
    ylim = 0,I.shape[1]


    Yg, Xg, Z1 = expand_matrix(list(Y), ylim, list(X), xlim, Z1) # put values everywhere for homogeneity (NB: here Z is defined with (Y,X) not (X,Y) be careful)
    Yg, Xg, Z2 = expand_matrix(list(Y), ylim, list(X), xlim, Z2) 

    if show_channels : 
        for number in numbers: # For each channel number 
            for l,letter in enumerate(letters) : 
                ax.text(X[l], Y[number-1], f'{letter}{number}', color = 'black')
    
    ax.imshow(I)
    
    ax = set_ax_parameters(ax, just_plot=True)

    # # Create the grid based on the computed X,Y values 
    # Xg,Yg = np.meshgrid(Xg,Yg)

    mapcolor1 = clm.Reds
    mapcolor1.set_under(color='white', alpha = 1)
    mapcolor1 = add_linear_transparency(mapcolor1)

    mapcolor2 = clm.Blues
    mapcolor2.set_under(color='white', alpha = 1)
    mapcolor2 = add_linear_transparency(mapcolor2)


    for_int = 10000

    step1 = int((for_int*max_z1 - for_int*min_z1)/steps_color)
    levels1 = np.array([i/for_int for i in range (int(for_int*min_z1), int(for_int*max_z1), step1)])
    cb1 = ax.contourf(Xg, Yg, Z1, cmap= mapcolor1, levels = levels1, antialiased=True, vmin = min_z1, vmax =max_z1,  algorithm = 'serial')
    
    step2 = int((for_int*max_z2 - for_int*min_z2)/steps_color)
    levels2 = np.array([i/for_int for i in range (int(for_int*min_z2), int(for_int*max_z2), step2)])
    cb2 = ax.contourf(Xg, Yg, Z2, cmap= mapcolor2, levels = levels2, antialiased=True,  vmin = min_z2, vmax =max_z2, algorithm = 'serial')
        
    # Show everything

    # We add a reference colorbar
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0)
    if yticks[0] is None :
        steptick1 = (int(for_int*max_z1) - int(for_int*min_z1))/5
        colorticks1= [round(i/for_int,3) for i in range (int(for_int*min_z1), int(for_int*max_z1)+int(steptick1), int(steptick1))]
    else :
        colorticks1 = yticks[0]
    cbar1 = plt.colorbar(cb1, cax=cax1, ticks = colorticks1)
    cbar1.ax.tick_params(labelsize= fontsize)
    cbar1.set_label('IILD frequency (Hz)', rotation=270, fontsize = 13, labelpad = 20, font = font)
    cbar1.ax.set_yticklabels(colorticks1)

    cax2 = divider.append_axes("left", size="5%", pad=0)
    if yticks[1] is None :
        steptick2 = (int(for_int*max_z2) - int(for_int*min_z2))/5
        colorticks2= [round(i/for_int,3) for i in range (int(for_int*min_z2), int(for_int*max_z2)+int(steptick2), int(steptick2))]
    else : 
        colorticks2 = yticks[1]
    cbar2 = plt.colorbar(cb2, cax=cax2, ticks = colorticks2)
    cbar2.ax.tick_params(labelsize= fontsize)
    cbar2.ax.yaxis.set_label_position('left')
    cbar2.ax.yaxis.tick_left()
    cbar2.set_label('MUA frequency (Hz)', rotation=90, fontsize = 13, labelpad = 20, font = font)
    cbar2.ax.set_yticklabels(colorticks2)

    if saving :
        plt.savefig(f'{folder_fig}\\{date_time}_{peak_criteria}.{format_save}', format = format_save)

    plt.show()

   
def make_boxplot(y, experiments_keys, namefig = 'plot', saving = False, draw_line = False, stars = [], limy = None, broken = None, fig = None, ax = None, show = True, yticks = None, ylabel = None, xticks = None, colors_plot = None) : 
    '''
    Creates boxplot based on values in y and a list of keys corresponding (y and experiments_keys have the same len)
    y : list of lists each list contains the values for each condition 
    experiments_keys : list of names of the conditions (with associated colors in colors_plot in the parameters_file)
    namefig: the name of the figure for saving 
    Put saving = True so the figure is saved 

    todo adjust the height of the stat bar so no overlap with dots 
    '''
    for i in range (len(y)) :
        print(f'{namefig} - {experiments_keys[i]}. Median : {np.nanmedian(y[i])}. Std : {np.nanstd(y[i])}, Mean: {np.nanmean(y[i])}, SEM: {sem(y[i])}')

    for star in stars :
        print(f'{experiments_keys[star[0]]} and {experiments_keys[star[1]]}, p = {star[3]}   ({star[4]})')
    
    # Get the right colors 
    if colors is not None : 
        try :
            colors_plot = [colors[exp] for exp in experiments_keys]
        except :
            colors_plot = ['white' for _ in experiments_keys]
    median_colors = [inverse_color[color] for color in colors_plot]
    
    y_box = [np.array(yi) for yi in y]
    y_box = [yi[~np.isnan(yi)].tolist() for yi in y_box]
    
    if ax is None :
        _, ax = start_fig(1, 1, figsize = fig_size)
    
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
    
    points = []
    for xi, yi,c in zip(x,y, colors_plot) :
        points.append(ax.scatter(xi, yi, color = c, edgecolors=edgecolor, s = cm2pts(dot_size)))


    for patch, color, median, median_color in zip(bp['boxes'], colors_plot, bp['medians'], median_colors):
        patch.set_facecolor(color)
        patch.set_linewidth(lw)
        median.set_color(median_color)
        median.set_linewidth(lw_median)

    if stars is not None : 
        ax = add_stats(ax, stars, limy)
    
    if broken is not None :
        add_broken_axis(fig, ax, [limy, broken], points = points, box = bp, boxcolors = colors_plot, stars = stars)
        

    if saving :
        namefig = namefig.replace(' ','_')
        plt.savefig(f'{folder_fig}\\{namefig}.{format_fig}', format = format_fig, transparent = True)
    
    if show : 
        plt.show()
    else : 
        return ax 

    print(f'{namefig} - y ticks : {ax.get_yticks()}')


## Very specific plot

def compare_peak_type_coloc(path, peak_type_ref, other_peak_type, peak_criteria = 'frequency', minimum = None, show = False) : 
    '''
    This can be useful for comparing the channels with and without each peak type but todo comment more 
    '''
    if minimum is None : 
        minimum = min_freq[peak_type_ref]
    date_time = date_time_from_path(path)
    channels = list_channels_all['sparse']
    channel_yes = []
    channel_no = []
    values = {peak_type : {} for peak_type in peak_types}
    channels, values = get_channels_and_z(date_time, peak_criteria, peak_type) 
    for channel in channels :
        for peak_type in peak_types : 
            values[peak_type][channel] = peak_info(date_time, channel, peak_type, peak_criteria)
            if peak_type == peak_type_ref : 
                if values[peak_type][channel] > minimum :
                    channel_yes.append(channel)
                else : 
                    channel_no.append(channel)


    box_yes = [values[other_peak_type][channel] for channel in channel_yes if values[other_peak_type][channel] > minimum]
    box_no = [values[other_peak_type][channel] for channel in channel_no if values[other_peak_type][channel] > minimum]

    if show : 
        make_boxplot([box_yes, box_no], [f'with {peak_type_ref}', 'without'], stars = [], saving = True)
    else : 
        return box_yes, box_no

## Useful for plots 


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
        if type(xticks[0]) == str :
            xticks_pos = [i for i in range(1,len(xticks)+1)]
        else : 
            xticks_pos = xticks 

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
                ax.set_xticks(xticks, xticks, font = font, fontsize = fontsize)
        else  :
            ax.set_xticks(xticks_pos, xticks, font = font, fontsize = fontsize)

        
        ax.tick_params(axis="x", length = 0)
        #ax.tick_params(axis="x",direction="in", length = cm2pts(tick_len), width = tick_thick) # ticks going in the plot at the right size 

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
                ax.set_yticks(yticks, yticks, font = font, fontsize = fontsize)
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

def add_linear_transparency(cmap) :
    from matplotlib.colors import ListedColormap
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)
    return my_cmap


    
def add_broken_axis(fig, ax, limits, points = None, box = None, boxcolors = None, stars = None):
    
    # Create two subplots and hide the spines between them
    axes = fig.axes 
    index = None

    for i in range (len(axes)) :
        try : 
            for j in range(len(axes[i])) :
                if axes[i][j] == ax :
                    index = i*len(axes[i])+j
        except :
            if axes[i] == ax :
                index = i 
    
    if index is None :
        print('AX NOT FOUND')
        fig, [ax1,ax2] = start_fig(ncols = 1, nrows = 2)
    
    else :
        nrows = ax.get_gridspec().nrows
        ncols = ax.get_gridspec().ncols
        ax1 = fig.add_subplot(nrows, ncols, index+1)
        pos = ax.get_position()
        ax1.set_position([pos.x0, pos.y0+4*pos.height/7,  pos.width, 3*pos.height/7])
        ax2 = ax
        ax2.set_position([pos.x0, pos.y0,  pos.width, 3*pos.height/7])
        ax2.set_ylabel('')


    # Set the limits of each subplot
    ax2.set_ylim(limits[0][0], limits[0][1])
    ax1.set_ylim(limits[1][0], limits[1][1])


    # Plot the data on both subplots
    lines = ax.get_lines()
    for line in lines  :
        ax1.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), alpha = line.get_alpha(), lw = line.get_linewidth())
        ax2.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), alpha = line.get_alpha(), lw = line.get_linewidth())

    if points is not None :
        for point in points :
            offsets = point.get_offsets()
            x = offsets[:,0]
            y = offsets[:,1]
            colors = point.get_facecolors()
            sizes = point.get_sizes()
            ax1.scatter(x,y, c = colors, s = sizes, edgecolor = edgecolor)
            ax2.scatter(x,y, c = colors, s = sizes, edgecolor = edgecolor)
    
    if box is not None :
        medians = box['medians']
        whisks = box['whiskers']

        ydata = []
        for y in whisks:
            ydata.append(y.get_ydata())

        n = 0
        for x in medians :
            minx, maxx = x.get_xdata()
            
            miny = ydata[2*n][0]
            maxy = ydata[2*n+1][0]

            ax1.add_patch(pt.Rectangle((minx,miny), maxx-minx, maxy-miny, facecolor = boxcolors[n], edgecolor = edgecolor))
            ax2.add_patch(pt.Rectangle((minx,miny), maxx-minx, maxy-miny, facecolor = boxcolors[n], edgecolor = edgecolor))

            n += 1

    # Add slanted lines to show the break in the axis
    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d,+d), (-d,+d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d,+d), (1-d,1+d), **kwargs)


    ax1 = set_ax_parameters(ax1, nticks=2, yticks = limits[1])
    ax2 = set_ax_parameters(ax2, nticks = 2, yticks = limits[0])
    
    ax2.spines['top'].set_visible(True)
    ax1.spines['bottom'].set_linestyle((0, (5, 10)))
    ax2.spines['top'].set_linestyle((0, (5, 10)))

    if stars is not None :
        ax1 = add_stats(ax1, stars, limits[1])

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


## PICKLE functions


def channel_from_pkl(date_time, file, ext = '.pkl') : 
    '''
    For a file named as whatever_date_time_channel.pkl, finds the channel name 
    '''
    index_channel = file.index(date_time)+len(date_time)+1
    index_ext = file.index(ext)
    channel = file[index_channel:index_ext]
    return channel 

def list_date_time_pkl(date_time) :
    '''
    Look in the peaks folder and get all the files, returns only the one with the right date_time in the name 
    ''' 
    list_analyzed = listdir(peaks_folder)
    list_analyzed = [file for file in list_analyzed if date_time in file]
    return list_analyzed



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




### MINOR USEFUL FUNCTIONS

def add_sheet(file:string, sheet_name:string) -> None: 
    '''
    Inputs: file is the path of the excel file we want to modify
            sheet_name is the name of the new excel sheet to put in
    No output
    '''
    wb = op.load_workbook(file)
    if sheet_name not in wb.sheetnames:
        wb.create_sheet(sheet_name)
    wb.save(file)


def print_duration(path) : 
    '''
    Find the total number of frames from h5 file 
    Input : path of interest
    No output but print the duration in minutes 
    '''
    minutes = get_minutes_duration(path)
    date_time = date_time_from_path(path)

    print(f'{date_time} : {minutes} minutes')


def expand_ID(start = start_row) : 
    '''
    This function changes the ID in ID_slice
    '''
    
    wb = op.load_workbook(info_file) 
    sheet = wb['conditions'] # Go to the right sheet 
    
    # Finds the columns of interest 
    for column in range (1,sheet.max_column) :
        if sheet.cell(row=1, column=column).value == 'ID' :
            column_ID = column 
        if sheet.cell(row=1, column=column).value in ['slice', 'Slice'] :
            column_slice = column 

    for row in range (start,sheet.max_row + 1) : 
        ID = sheet.cell(row=row, column=column_ID).value # Get the exp ID
        slice = sheet.cell(row=row, column=column_slice).value # Get the slice number
        sheet.cell(row=row, column=column_ID).value = f'{ID}_{slice}' # change the value
    
    # save the file
    wb.save(info_file)

def expand_matrix(index_i, lim_i, index_j, lim_j, values, filler = 0) : 
    '''
    Put 0 (default) or other filler value in a matrice to expand it to lim_i and lim_j (used in colormap so that the 0 values are everywhere on the picture for an homogeneous background)
    '''
    if lim_i[0] < index_i[0] :
        index_i = [lim_i[0]] + index_i 
        values = [[filler]*len(index_j)] + values

    if lim_i[1] > index_i[-1] :
        index_i = index_i + [lim_i[1]]
        values = values + [[filler]*len(index_j)]

    if lim_j[0] < index_j[0] :
        index_j = [lim_j[0]] + index_j
        for v in range(len(values)) :
            values[v] = values[v] + [filler]

    if lim_j[1] > index_j[-1] :
        index_j = index_j + [lim_j[1]]
        for v in range(len(values)) :
            values[v] = [filler] + values[v]

    return index_i, index_j, values 

def rounding(list) : 
    '''
    I created this function for making prettier list of values 
    todo This is not optimal. 
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

def div (x, d = 10) : 
    m = x //d
    k = x - d*(m)
    return (m,k)

## TIMING functions TO UPDATE

def time_limiting(y) :
    '''
    Need to update this as part of time_evol
    '''

    def end(y,duration) :
        return y[len(y)-duration:]

    def begin(y,duration) :
        return y[:duration]

    def random(y,duration) : 
        random_start = randint(0,len(y)-duration)        
        return y[random_start:random_start+duration]

    len_conditions = [[] for _ in range (len(y[0]))]
    for exp in y :
        for c,condition in enumerate(exp) : 
            len_conditions[c].append(len(condition))
    min_len = [min(len_cond) for len_cond in len_conditions]
    f = [[] for _ in range (len(y))]
    for exp in range (len(y)) :
        f[exp] = [[] for _ in range (len(y[exp]))]
        for condition in range (len(y[exp])) :
            f[exp][condition] = end(y[exp][condition], min_len[condition])
    return f 


def plot_in_time (y, conditions, only_median = True, put_rectangles = True, limy = None) : 
    ''''
    Inputs :
    y: list of results for each experiments (dimension = 3), y[n][c][t] is the value at a specific time point t for the condition c for the experiment n 
    conditions: list of conditions (len = 2nd dimension of y)
    time_limit: True/False using a limited duration defined in the parameters file 
    only_median: Put True to only have one plot with the median values for all experiments // False : individual plot for each experiment 
    put_rectangles: with True, the different conditions are distinguished by rectangles (color defined in the parameters file)
    limy: a specific range for the y axis to be respected 

    todo update this!!
    '''
    colorplot = [colors[condition] for condition in conditions]

    def range_cumul(y) : 
        startstop = [0]
        startstop.extend([len(y[i]) for i in range(len(y))])
        startstop = np.cumsum(startstop)
        return [list(range(startstop[i]-i,startstop[i+1]-i)) for i in range (len(y))]

    def rectangling(ax, limit, color) : 
        # The height of the rectangles depends on the y axis 
        vertical = ax.get_ylim()
        # For each rectangle to draw
        # rectangle is defined as: (origin x, origin y), size horizontal, size vertical (starting from bottom left corner)
        Rect = plt.Rectangle((limit[0], vertical[0]), limit[1]-limit[0], vertical[1]-vertical[0], alpha = alpha_plots, color = color)
        # draw the created rectangle 
        ax.add_patch(Rect)


    def separate_plot (y) :
        for i in range (len(y)) :
            _, ax = start_fig()
            range_val = range_cumul(y[i])
            for j in range (len(y[i])) :
                ax.plot(range_val[j],y[i][j], color = 'black')
                if limy is not None:
                    ax.set_ylim(limy)
                if put_rectangles :
                    rectangling(ax, [range_val[j][0], range_val[j][-1]], colorplot[j])     
            plt.show()

    def median_plot(f) :
        _, ax = start_fig()
        f = np.nanmedian(f, axis = 0)
        range_val = range_cumul(f)
        for j in range (len(f)) :
            ax.plot(range_val[j],f[j], color = 'black')
            if limy is not None:
                ax.set_ylim(limy)
            if put_rectangles :
                rectangling(ax, [range_val[j][0], range_val[j][-1]], colorplot[j])
        plt.show() 

    if only_median :
        median_plot(y)
    else :
        separate_plot(y)

def freq_all_from_exp_ID(check, peak_type, channels_to_use = 'best_baseline', return_channels = False, excel = False) : 
    '''
    This should be part of y_from_exp_ID as a big class that gives all the relevant info based on exp_ID..
    todo comment more and update it 
    '''

    if excel :
        get_peaks = get_peaks_from_excel
    else :
        get_peaks = get_peaks_from_pkl


    ID_dict = exp_ID_dict() # See funtion description
    exp_valid = [exp_ID for exp_ID in ID_dict.keys() if check['condition'] in ID_dict[exp_ID][check['category']].keys() and ID_dict[exp_ID][check['category']][check['condition']] == check['criteria']]

    frequencies = [[] for _ in range (len(exp_valid))] # Creates a list to store the frequencies


    for e,exp_ID in enumerate(exp_valid) : # For each experiment (each slice is considered independent generally)

        frequencies[e] = [[] for _ in range (len(ID_dict[exp_ID]['date_times']))]
        
        for d, date_time in enumerate(ID_dict[exp_ID]['date_times']) :
            # Find the time 
            
            start, stop, bad = frames_to_use(date_time) # in frames 

            if start is None :
                start = 0

            if stop is not None and start + after_time + duration > stop :
                print(f'ERROR in duration for {date_time}, start = {(stop-duration)/(60*freqs)} min instead of {(start+after_time)/(60*freqs)}')
                start = stop - duration 
            else :
                start += after_time
                stop = start + duration

            if date_time in ['2023-01-12_16-01-55', '2023-01-12_16-01-56', '2023-01-12_16-01-57'] : 
                date_time = '2023-01-12_16-01-57'

            if channels_to_use == 'best_baseline' and d == 0 : 
                channels = channels_ordered_from_results(date_time, peak_type)[:10]
                
            if channels_to_use == 'all' : 
                channels = channels_ordered_from_results(date_time, peak_type)
            
            if type(channels_to_use) == list : 
                channels = channels_to_use

            frequencies[e][d] = [[] for _ in range (len(channels))]
            
            if d == 0 :
                freq = [0 for _ in range (len(channels))]

            for c,channel in enumerate(channels) :

                frequencies[e][d][c] = [0 for _ in range(int(start),int(stop),int(30*freqs))]
                
                
                frames, _,_ = get_peaks(channel, date_time, selected_peaks = False)

                frames = frames[peak_type]

                if frames != [] and not np.isnan(frames).all(): 
                    
                    indices = [i for i in range (len(frames)) if frames[i] >= start and frames[i] < stop]
                    todel = []
                    for j in range (len(bad)) :
                        for i in indices :
                            if i in range(bad[j]-5*freqs,bad[j]+5*freqs) : 
                                todel.append(i)
                    indices = [i for i in indices if i not in todel]
                    
                    if d == 0 :
                        freq[c] = freqs*len(indices)/duration


                    if len(indices) > 0 : 
                        k = 0 
                        for t in range(int(start),int(stop),int(30*freqs)) : 
                            ind = [frames[i] for i  in indices if frames[i]>= t and frames[i] < t+30*freqs]
                            frequencies[e][d][c][k] = len(ind)*freqs/(30*freqs)
                            k += 1
                        
            if channels_to_use == 'best_baseline' and d == 0  :
                best_channels_baseline = list(np.argsort(freq))
                best_channels_baseline.reverse()
                best_channels_baseline = [chan for chan in best_channels_baseline if not np.isnan(freq[chan])][:N_best]
                channels = [channels[chan] for chan in best_channels_baseline]
                frequencies[e][d] = [frequencies[e][d][chan] for chan in best_channels_baseline]


    if return_channels :
        return frequencies,  channels
    else :
        return frequencies
     
 
def time_evol(path, channel, peak_type, period = 60*freqs) : 
    '''
    this is to make lists of frequencies and amplitudes in time, each point of the list is a median of the freq/amp in the period 
    todo: make this better, see final_plotting etc
    '''
    raw, channels = raw_signals_from_file(path)
    channel_index = channels.index(channel)
    # Take only the signal from the channel chosen
    raw_channel = raw[channel_index]

    # Call the class preprocessing to filter, (denoise) and normalize the signal 
    data = preprocessing(raw_channel, peak_types)
    # Call the class find_events to find the peaks for each defined peak type 
    peaks = find_events(data)
    
    frames = peaks.frame_index[peak_type]
    amplitudes = peaks.amplitude[peak_type]['raw']
    
    freq = []
    amp = []

    for t in range (0, max(frames), period) :
        good_i = [i for i in range (len(frames)) if frames[i]>= t and frames[i] < t + period]
        frames_time = [frames[i] for i in good_i]
        amplitudes_time = [factor_amp*abs(amplitudes[i]) for i in good_i]
        freq.append(freqs*len(frames_time)/period)
        amp.append(np.nanmedian(amplitudes_time))

    return freq,amp


## Others


def check_for_problems() :
    '''
    todo : change my code so I take this into account at the beginning, I discovered that I get my data in the unit defined in the InfoChannel, with the exponent defined there
    I have to multiply my data by a conversion factor.. Weird
    I did this to check if they all have the same units, exponent etc but this is not ideal, change this!!!
    '''

    problems = {}
    infos_interest = {'unit' : {'index' : 6, 'val' : b'V'}, 'exponent' : {'index' : 7, 'val' : -12}, 'step' : {'index' : 8, 'val' : 0}, 
                    'freq' : {'index' : 9, 'val' : 100}, 'factor' : {'index' : 10, 'val' : 59605}}

    paths = choose_files_from_info()
    for path in paths :
        problems[path] = []
        data = h5py.File(path)
        k = 0 
        while True :
            try :
                data = data['Data']['Recording_0']['AnalogStream'][f'Stream_{k}']
                break
            except :
                k += 1
        # Get the channels names
        data = data['InfoChannel'] 
        problems[path] = [[] for _ in range (len(data))]
        for c, channel in enumerate(data) : 
            for key in infos_interest.keys() : 
                if channel[infos_interest[key]['index']] != infos_interest[key]['val'] :
                    text = f"{key} : {channel[infos_interest[key]['index']]}"
                    print(text)
                    problems[path][c].append(text)
