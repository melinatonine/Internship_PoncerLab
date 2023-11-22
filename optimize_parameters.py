## Ideas

## For each peak type 
# 1) User_find_peaks - look at 30 seconds of 5 recordings with IILDs/MUA and click on the peaks / and 5 without
# 2) Save the raw data and the frames chosen
# 3) For filter orders between 1 and 10, for thresholds between 1 and 15 : look at precision and sensitivity
#       precision: for each frame found, if exist frame user such as dist(frame found, frame user) < 250ms : p += 1 
#                   -> p/(N frames found)
#       sensitivity: for each frame user, if exist frame found such as dist(frame found, frame user) < 250ms : s += 1 
#                    -> s/(N frames user)

from MEA_functions import *
from specific_parameters import *
from general_parameters import *
import random 

folder_save = r'W:\Analysis\MEA\optimization\\'
distances_max = {'interictal' : 0.1*freqs, 'MUA' : 0.02*freqs}
filters_train = {peak_type : list(range(1,10)) for peak_type in peak_types}
thresholds_train = {peak_type : list(range(1,20)) for peak_type in peak_types}

# list of recordings with IILDs, MUA 

def save_files_interest () : 
    recordings_with = {'interictal' : ['2023-04-25_13-58_F7', '2023-05-23_14-32_B6', '2023-05-23_22-08_F3', '2023-10-03_17-28_C5', '2023-10-10_13-42_H10'], 'MUA' : ['2023-04-06_19-12_H7', '2023-05-09_17-14_K4', '2023-05-23_17-34_E10', '2023-10-03_14-51_H7', '2023-10-10_16-00_J8']}
    files_interest = {}
    for peak_type in peak_types : 
        files_interest[peak_type] = {}
        for recording in recordings_with[peak_type] : 
            folder = r'W:\DATA\MEA\\' + recording[:10] + r'\\'
            list_files = listdir(folder)
            list_files = [r'{}'.format(file) for file in list_files if '.h5' in file]
            h = recording[11:13]
            list_good_hour_files = [file for file in list_files if file[11:13] == h]
            minute = recording[14:16]
            dist_minutes = [abs(int(file[14:16]) - int(minute)) for file in list_good_hour_files]
            file = list_good_hour_files[np.argmin(dist_minutes)]
            path = folder + '\\' + file 
            date_time = date_time_from_path(path)
            channel_with = recording[17:]
            channel_without = list_channels_all['sparse'][np.mod(list_channels_all['sparse'].index(channel_with) + 60, len(list_channels_all['sparse'])-1)] 
            if channel_without == 'M7' :
                channel_without = 'M2'
            files_interest[peak_type][date_time] = {'path' : path, 'channels' : [channel_with, channel_without]}
    save_dict(files_interest, f'{folder_save}file_list.pkl')

#save_files_interest()

def save_raw(duration = {'interictal' : 3*60*freqs, 'MUA' : 20*freqs}) : 
    files_interest = read_dict(f'{folder_save}file_list.pkl')
    for peak_type in peak_types :
        for date_time in files_interest[peak_type].keys() :
            print(f'{date_time} processing                ', end = '\r')
            path = files_interest[peak_type][date_time]['path']
            raw, channels = raw_signals_from_file(path)
            for channel in files_interest[peak_type][date_time]['channels'] :
                if channel in channels : 
                    raw_channel = raw[channels.index(channel)]
                    frame_start = random.randint(0, len(raw_channel)- duration[peak_type])
                    raw_channel = raw_channel[frame_start : frame_start + duration[peak_type]]
                    raw_channel = factor_amp*raw_channel
                    data = {'raw' : raw_channel, 'frame_start' : frame_start, 'duration' : duration[peak_type]}
                    save_dict(data, f'{folder_save}{date_time}_{channel}.pkl')

#save_raw()       

def choose_peaks(rewrite = False, window = {'interictal' : 10*freqs, 'MUA' : int(2.5*freqs)}) :

    def file_to_frames(folder_save, date_time, channel) :
        file = f'{folder_save}{date_time}_{channel}.pkl'
        data = read_dict(file)
        raw = data['raw']
        d = data_time_window(raw, window = window[peak_type], ylim = [-100,100], get_frames=True)
        save_dict(d.list_x, f'{folder_save}{date_time}_{channel}_peaks.pkl')

    files_interest = read_dict(f'{folder_save}file_list.pkl')
    for peak_type in peak_types :
        date_times = list(files_interest[peak_type].keys())
        gui_dict = {}
        for date_time in date_times : 
            gui_dict[date_time] = {'label' : date_time, 'var' : [],'value' : 0}
        gui_dict = make_choices(gui_dict, text_keys = [], checkbox_keys = date_times)
        date_times_chosen = []
        for date_time in date_times :
            if gui_dict[date_time]['value'] == 1 :
                date_times_chosen.append(date_time)
        for date_time in date_times_chosen :
            for channel in files_interest[peak_type][date_time]['channels'] :
                if not rewrite : 
                    try :
                        _ = read_dict(f'{folder_save}{date_time}_{channel}_peaks.pkl')
                        print('already done')
                    except : 
                        file_to_frames(folder_save, date_time, channel)            
                else :
                    file_to_frames(folder_save, date_time, channel) 

#choose_peaks(rewrite = False)


def find_peaks_simplified(peak_type, raw, filter_order, threshold_fact) :
    b,a = signal.butter(filter_order, param[peak_type]['filter_freq'], btype = param[peak_type]['filter_type'], fs = freqs)
    filtered = signal.filtfilt(b, a, raw) 
    absolu = np.abs(filtered)
    threshold = threshold_fact*np.median(np.absolute(absolu - np.median(absolu)))
    frames, _ = signal.find_peaks(absolu, height = threshold, distance = param[peak_type]['inter_event'], width = param[peak_type]['peak_duration'])
    return frames


def computer_peaks() : 
    files_interest = read_dict(f'{folder_save}file_list.pkl')
    for peak_type in peak_types :
        for date_time in files_interest[peak_type].keys() :
            for channel in files_interest[peak_type][date_time]['channels'] :
                print(f'{date_time}_{channel}', end = '\r')
                data = read_dict(f'{folder_save}{date_time}_{channel}.pkl')
                raw = data['raw']
                frames_computer = {}
                for f in filters_train[peak_type] : 
                    for t in thresholds_train[peak_type] : 
                        frames_computer[(f,t)] = find_peaks_simplified(peak_type, raw, f, t)
                        save_dict(frames_computer, f'{folder_save}{date_time}_{channel}_comput.pkl')

#computer_peaks()

def train_test() :
    files_interest = read_dict(f'{folder_save}file_list.pkl')
    data_test_train = {}
    for peak_type in peak_types : 
        data_test_train[peak_type] = {}
        data_test_train[peak_type]['test'] = [list(files_interest[peak_type].keys())[random.randint(0,len(list(files_interest[peak_type].keys()))-1)]]
        data_test_train[peak_type]['train'] = [date_time for date_time in files_interest[peak_type].keys() if date_time not in data_test_train[peak_type]['test']]
    save_dict(data_test_train, f'{folder_save}train_test.pkl')

#train_test()

def count_F1_score(distances_max = {'interictal' : 0.1*freqs, 'MUA' : 0.02*freqs}, category = 'train', list_f = filters_train, list_t = thresholds_train) : 
    date_times = read_dict(f'{folder_save}train_test.pkl')
    files_interest = read_dict(f'{folder_save}file_list.pkl')
    F1 = {}
    f_t = {}
    for peak_type in peak_types :
        F1[peak_type] = []
        f_t[peak_type] = []
        for f in list_f[peak_type] : 
            for t in list_t[peak_type] : 
                tp = 0
                s_div = 0
                p_div = 0
                for date_time in date_times[peak_type][category] :
                    for channel in files_interest[peak_type][date_time]['channels'] :
                        frames_user = read_dict(f'{folder_save}{date_time}_{channel}_peaks.pkl')
                        frames_computer = read_dict(f'{folder_save}{date_time}_{channel}_comput.pkl')
                        frames_computer = frames_computer[(f,t)]

                        if len(frames_user) > 0 : 
                            for frame_u in frames_user :
                                distances = [abs(frame_u - frame_c) for frame_c in frames_computer]
                                if len(distances) > 0 : 
                                    if min(distances) < distances_max[peak_type] : 
                                        tp += 1

                        s_div += len(frames_user)
                        p_div += len(frames_computer)


                if s_div == 0 : 
                    if p_div != 0 :
                        F1[peak_type].append(tp/p_div)
                        f_t[peak_type].append((f,t))
                elif p_div == 0 :
                        F1[peak_type].append(tp/s_div)
                        f_t[peak_type].append((f,t))
                else : 
                    F1[peak_type].append((tp/s_div+tp/p_div)/2)
                    f_t[peak_type].append((f,t))
    
    save_dict((F1,f_t), f'{folder_save}F1_f_t_{category}.pkl')

#count_F1_score()

def compare_peaks(date_time, channel, peak_type, f, t) : 
    window = {'interictal' : 10*freqs, 'MUA' : int(2.5*freqs)}
    folder_save = r'W:\Analysis\MEA\optimization\\'
    frames_computer = read_dict(f'{folder_save}{date_time}_{channel}_comput.pkl')
    frames_computer = frames_computer[(f,t)]
    frames_user = read_dict(f'{folder_save}{date_time}_{channel}_peaks.pkl')
    true_pos = []
    false_pos = []
    false_neg = []
    if len(frames_user) > 0 : 
        for frame_u in frames_user :
            distances = [abs(frame_u - frame_c) for frame_c in frames_computer]
            if len(distances) > 0 : 
                if min(distances) < distances_max[peak_type] : 
                    true_pos.append(frame_u)
                else : 
                    false_neg.append(frame_u)

    if len(frames_computer) > 0 : 
        for frame_c in frames_computer :
            distances = [abs(frame_u - frame_c) for frame_u in frames_user]
            if len(distances) > 0 : 
                if min(distances) >= distances_max[peak_type] : 
                    false_pos.append(frame_c)

    file = f'{folder_save}{date_time}_{channel}.pkl'
    data = read_dict(file)
    raw = data['raw']
    data_time_window(raw, window = window[peak_type], ylim = [-100,100], frames_to_show={'green': true_pos, 'red' : false_pos, 'grey' : false_neg})


def test() : 
    folder_save = r'W:\Analysis\MEA\optimization\\'
    date_times = read_dict(f'{folder_save}train_test.pkl')
    files_interest = read_dict(f'{folder_save}file_list.pkl')
    F1, f_t = read_dict( f'{folder_save}F1_f_t_train.pkl')
    best = {}
    for peak_type in peak_types : 
        F1[peak_type] = [f1 for f1 in F1[peak_type] if f1 != np.nan]
        best[peak_type] = f_t[peak_type][np.argmax(F1[peak_type])]
        print(f'For {peak_type}, train data, best filter order is {best[peak_type][0]} and best threshold is {best[peak_type][1]}, with a F1 score of {np.max(F1[peak_type])}')
        date_time = date_times[peak_type]['test'][0]
        channel = files_interest[peak_type][date_time]['channels'][0]
        compare_peaks(date_time, channel, peak_type, f=best[peak_type][0], t=best[peak_type][1])
    count_F1_score(category='test', list_f = {peak_type : [best[peak_type][0]] for peak_type in peak_types}, list_t = {peak_type : [best[peak_type][1]] for peak_type in peak_types})
    F1, f_t = read_dict( f'{folder_save}F1_f_t_test.pkl')
    for peak_type in peak_types : 
        print(f'For {peak_type}, with filter {best[peak_type][0]} and threshold {best[peak_type][1]}, test F1 score is {(F1[peak_type])}')
        
    
test()