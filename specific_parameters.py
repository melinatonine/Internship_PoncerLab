class get_param :
    def __init__(self, name) :

        folder = r'W:\Analysis\MEA\\' + name + r'\\'
        self.files = folder

        self.infos = folder + r'infos.xlsx'
        self.figures = folder + r'figures\\'
        self.positions = folder + r'positions\\'
        
        self.peaks = folder  + r'peaks\\'
        self.selected_peaks = folder + r'selected_peaks'
        self.results =  folder  + r'results\\'
        self.param =  folder  + r'param\\'

        if name == 'task3' : 
            self.proportion_slices_IILDs = {'SH-2022-002' : {'yes':[5,6,7,9], 'no' : [3,4,8]}, 'SH-2022-004' : {'yes' : [5,7,8], 'no' : [1,2,3,4,6,9,10]}, 'SH-2022-005' : {'yes' : [2,3,5,6], 'no' : [1,4,7,8,9,10]}, 'SH-2023-000' : {'yes' : [2,3], 'no' : [1,4]}, 'SH-2023-001' : {'yes': [3,4,5,6,8,9], 'no' : [10,7]}}

            specials = {}
            specials['2023-01-12_16-01-55'] = {'start_stop' : [300, 1560], 'path' : r'W:\DATA\MEA\2023-01-12\2023-01-12T16-01-57McsRecording.h5'}
            specials['2023-01-12_16-01-56'] = {'start_stop' : [1560,2220], 'path' : r'W:\DATA\MEA\2023-01-12\2023-01-12T16-01-57McsRecording.h5'}
            specials['2023-01-12_16-01-57'] = {'start_stop' : [2220,2820], 'path' : r'W:\DATA\MEA\2023-01-12\2023-01-12T16-01-57McsRecording.h5'}
            specials['2023-01-12_16-01-58'] = {'start_stop' : [300,1560], 'path' : r'W:\DATA\MEA\2023-01-12\2023-01-12T16-01-57McsRecording.h5'}
            specials['2023-01-12_16-01-59'] = {'start_stop' : [1560,2220], 'path' : r'W:\DATA\MEA\2023-01-12\2023-01-12T16-01-57McsRecording.h5'}
            specials['2023-08-08_16-15-17'] = {'start_stop' : [0,420], 'path' : r'W:\DATA\MEA\2023-08-08\2023-08-08T16-15-17McsRecording.h5'}
            specials['2023-08-08_16-15-18'] = {'start_stop' : [420,900], 'path' : r'W:\DATA\MEA\2023-08-08\2023-08-08T16-15-17McsRecording.h5'}
            specials['2023-08-08_16-15-19'] = {'start_stop' : [900,1620], 'path' : r'W:\DATA\MEA\2023-08-08\2023-08-08T16-15-17McsRecording.h5'}
            specials['2023-08-08_17-23-22'] = {'start_stop' : [0,840], 'path' : r'W:\DATA\MEA\2023-08-08\2023-08-08T17-23-22McsRecording.h5'}
            specials['2023-08-08_17-23-23'] = {'start_stop' : [840,1620], 'path' : r'W:\DATA\MEA\2023-08-08\2023-08-08T17-23-22McsRecording.h5'}
            specials['2023-08-08_17-23-24'] = {'start_stop' : [1620,None], 'path' : r'W:\DATA\MEA\2023-08-08\2023-08-08T17-23-22McsRecording.h5'}
            specials['2023-08-08_18-37-08'] = {'start_stop' : [0,660], 'path' : r'W:\DATA\MEA\2023-08-08\2023-08-08T18-37-08McsRecording.h5'}
            specials['2023-08-08_18-37-09'] = {'start_stop' : [660,1680], 'path' : r'W:\DATA\MEA\2023-08-08\2023-08-08T18-37-08McsRecording.h5'}
            specials['2023-08-08_18-37-10'] = {'start_stop' : [1680,None], 'path' : r'W:\DATA\MEA\2023-08-08\2023-08-08T18-37-08McsRecording.h5'}
            specials['2023-04-25_18-11-01'] = {'start_stop' : [0,1590], 'path' : r'W:\DATA\MEA\2023-04-25\2023-04-25T18-11-01McsRecording.h5'}
            self.specials = specials 

            self.to_highlight = ['SH-2023-001_4', 'SH-2023-001_6', 'SH-2023-001_5']
                    
            ## Time info for 3 conditions 
            freqs = 10000
            drugs = 'terbinafine', 'terbinafine_100', 'chet3', 'NPBA_10', 'NPBA_20'
            self.timing = {'baseline' : {'type' : 'end', 'time' : 30*freqs, 'duration' : 3.5*60*freqs}, 'washout' : {'type' : 'after', 'time' : 5*60*freqs, 'duration' : 3.5*60*freqs}}
            for drug in drugs : 
                self.timing[drug] = self.timing['washout'].copy()

            self.n_conditions = 3
            self.N_best = 1

            # PLOT colors 
            self.colors = {'baseline' : 'grey', 'terbinafine' : 'cadetblue', 'NPBA_10' : 'goldenrod', 'NPBA_20' : 'goldenrod','washout' : 'grey', 'terbinafine_100' : 'darkcyan', 'chet3' : 'chartreuse'}
            self.inverse_color = {'black': 'white', 'white' : 'black', 'grey':'black', 'goldenrod':'black', 'cadetblue':'black', 'darkcyan':'black', 'chartreuse' : 'black'}

            self.drugs = ['terbinafine', 'terbinafine_100', 'NPBA_10', 'chet3']

            

        if name == 'ictal' : 
            
            ## Time info for 3 conditions 
            freqs = 10000
            self.timing = {'baseline' : {'type' : 'end', 'duration' : 4*60*freqs, 'time' : 0}, 'baseline+' : {'type' : 'end', 'duration' : 4*60*freqs, 'time' : 0}, 'washout' : {'type' : 'end', 'duration' : 4*60*freqs, 'time' : 0}, 'washout+' : {'type' : 'end', 'duration' : 4*60*freqs, 'time' : 0}, 
                           'baseline_drug_washout' : {'type' : 'after', 'time': 0, 'duration' : 5*60*freqs}, 'CLP' : {'type' : 'end', 'duration' : 4*60*freqs, 'time' : 0}}

            self.n_conditions = 2
            self.N_best = 1

            self.specials = {}

            # PLOT colors 
            self.colors = {'baseline' : 'grey', 'proconvulsant' : 'goldenrod'}
            self.inverse_color = {'black': 'white', 'white' : 'black', 'grey':'black', 'goldenrod':'black'}






