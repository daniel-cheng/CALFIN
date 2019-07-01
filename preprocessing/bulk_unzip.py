# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:25:54 2019

@author: Daniel
"""

import os, glob, shutil, tarfile
  
steps = ['1', '2']
steps = ['1']

if '1' in steps:
    source = r"D:/Daniel/Documents/Github/CALFIN Repo/downloader/rasters/Landsat/Greenland/zipped"
    unzip_dest = r"D:/Daniel/Documents/Github/CALFIN Repo/downloader/rasters/Landsat/Greenland/"
    move_dest = r"F:/Daniel/Documents/Github/CALFIN Repo/downloader/rasters/Landsat/Greenland/zipped"
    dry_run = False
    # domains_to_move = ['Hayes', 'Kangiata-Nunata', 'Kong-Oscar', 'Rink-Isbrae']
	domains_to_move = ['Helheim']
    
    # Function to rename multiple files 
    for domain in os.listdir(source):
        #Make unzip domain folder if not existing already
        unzip_domain_path = os.path.join(unzip_dest, domain)
        if not os.path.exists(unzip_domain_path):
            os.mkdir(unzip_domain_path)
            
        #Make move domain folder if not existing already, only if we want to move them
        if domain in domains_to_move:
            move_domain_path = os.path.join(move_dest, domain)
            if not os.path.exists(move_domain_path):
                os.mkdir(move_domain_path)
        
        #For each file in domain folder, unzip to appropiate unzip destination/year.
        for source_path in glob.glob(source + '/' + domain + '/*'): 
            filename = source_path.split(os.path.sep)[-1]
            basename = filename.split('.')[0]
            basename_parts = basename.split('_')
            
            satellite = basename_parts[0]
            level = basename_parts[1]
            path_row = basename_parts[2]
            date = basename_parts[3]
            processing_date = basename_parts[4]
            collection_number = basename_parts[5]
            tier = basename_parts[6]
            
            year = date[0:4]
            month = date[4:6]
            day = date[6:8]
            
            path = path_row[0:3]
            row = path_row[3:6]
            
            band = ''
            if satellite == 'LC08':
                band = 'B5'
            elif satellite == 'LE07':
                band = 'B4'
            elif satellite == 'LT05':
                band = 'B4'
            elif satellite == 'LT04':
                band = 'B4'
            elif satellite == 'LM05':
                band = 'B4'
            elif satellite == 'LM04':
                band = 'B4'    
            elif satellite == 'LM03':
                band = 'B7'
            elif satellite == 'LM02':
                band = 'B7'
            elif satellite == 'LM01':
                band = 'B7'
            else:
                band = 'B5'
                
            unzip_domain_year_path = os.path.join(unzip_domain_path, year)
            if not os.path.exists(unzip_domain_year_path):
                os.mkdir(unzip_domain_year_path)
            
            dest_basename = '_'.join([satellite, level, year + '-' + month + '-' + day, path + '-' + row, tier])
            bNIR_source_filename = basename + '_' + band + '.TIF'
            bNIR_dest_filename = '_'.join([dest_basename, band + '.TIF'])
            bNIR_temp_dest_file_path = os.path.join(unzip_domain_year_path, bNIR_source_filename)
            bNIR_final_dest_file_path = os.path.join(unzip_domain_year_path, bNIR_dest_filename)
            
            bQA_source_filename = basename + '_BQA.TIF'
            bQA_dest_filename = '_'.join([dest_basename, 'BQA.TIF'])
            bQA_temp_dest_file_path = os.path.join(unzip_domain_year_path, bQA_source_filename)
            bQA_final_dest_file_path = os.path.join(unzip_domain_year_path, bQA_dest_filename)
            
            if not os.path.exists(bNIR_final_dest_file_path) or not os.path.exists(bQA_final_dest_file_path):
                print('Extracting:', source_path, 'to:', bNIR_final_dest_file_path, bQA_final_dest_file_path)
                if dry_run == False:
                    try:
                        t = tarfile.open(source_path, 'r')
                        bNIR_member = t.getmember(bNIR_source_filename)
                        bQA_member = t.getmember(bQA_source_filename)
                        t.extractall(unzip_domain_year_path, members=[bNIR_member, bQA_member])
                        shutil.move(bNIR_temp_dest_file_path, bNIR_final_dest_file_path)
                        shutil.move(bQA_temp_dest_file_path, bQA_final_dest_file_path)
                        t.close()
                        print('Success!')
                    except:
                        print('Error in tarfile')
                        try:
                            t.close()
                        except:
                            print('unable to close tarfile')
                        continue
    
                if domain in domains_to_move:
                    move_file_path = os.path.join(move_domain_path, filename)
                    print(move_file_path)
                    if dry_run == False:
                        shutil.move(source_path, move_file_path)

if '2' in steps:
    source = r'D:/Daniel/Pictures/CALFIN/test_full/'
    dest = r'D:/Daniel/Pictures/CALFIN/test_all/'
    dry_run = 1
    
    # Function to rename multiple files 
    for domain_path in glob.glob(source + '*'):
        for filename in os.listdir(domain_path):
            source_path = os.path.join(domain_path, filename)
            dest_path = os.path.join(dest, filename)
#            print(source_path, dest_path)
            if dry_run != 0:
                shutil.copy(source_path, dest_path)