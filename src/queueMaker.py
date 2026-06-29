####
import pandas as pd
import numpy as np
from datetime import date
####

def generateMSQueue (instrument_type, 
                     rowID, 
                     colID,
                     rackID,
                     MSmethod ='some', 
                     LCmethod = 'sciex', 
                     instName = 'OA', 
                     instNo = 4, 
                     evosepNo = 11, 
                     gradientID = 'gradient', 
                     personalID = 'DeOl', 
                     expID = 'DO066_1_A032', 
                     ThermoMethodPath = 'C:\\Xcalibur\\methods\\Tune1p1\\Denys\\', 
                     number_of_blanks = 2,
                     blankRackID = 2, 
                     byRow = False,
                     full = False,
                     add_blanks = False, 
                     randomize = False,
                     output_path = r'D:\\ThermoQueues\\',
                     output_name = 'queue.csv',
                     preview = False):
    
    """
    Generates an MS acquisition queue, compatible with Thermo and Sciex instruments.
    :param str instrument_type: A class of an instrument of interest. Takes only names Thermo or Sciex. 
    :param list rowID: a list of desired rows in 96-well plate format (i.e. A, B, C, D)
    :param list colID: a list of desired columns in 96-well plate format (i.e. 1, 2, 3, 4)  
    :param int rackID: a number of rack of EvoSep LC (i.e. 1, 2, 3, 4, 5 or 6)
    :param str MSmethod: a desired MS method
    :param str LCmethod: a desired MS method. Required only for Sciex acquisition queue.
    :param str instName: an MS instrument ID.
    :param int instNo: a number of MS
    :param int evosepNo: a number of LC
    :param str gradientID: a type of LC gradient being used
    :param str personalID: personal ID
    :param str expID: ID of an existing experiment
    :param str ThermoMethodPath: a path to the MS method on the Thermo MS computer
    :param int number_of_blanks: how many blank runs should be add in the beginning and end of the queue
    :param int blankRackID: a number of rack of EvoSep LC for blanks (i.e. 1, 2, 3, 4, 5 or 6)
    :param bool full: is a full EvoSep box used? 
    :param bool randomize: should the sample order be randomized? 
    :param bool add_blanks: should blank runs be added?
    :param bool byRow: should the queue be ordered by row (i.e. A1, A2, A3 ...) or by column (i.e. A1, B1, C1 ...)?
    :param str output_path: path of the output dataset
    :param str output_name: name of the output file
    :param bool preview: should the dataset be previewed?
    
    Example use:
    if preview == False:
    
    generateMSQueue(instrument_type = 'Thermo', rowID = ['A','B'], colID = [1,2], rackID = 3, MSmethod = 'desired_method', LCmethod = 'desired_method', instName = 'OA', instNo = 4, evosepNo = 11, 
    gradientID = 'desired_gradient', personalID = 'DeOl', expID = 'experiment', ThermoMethodPath = 'path', number_of_blanks = 2, blankRackID = 2, byRow = False, full = False, add_blanks = True, randomize = True,
    output_path = 'desired_path', output_name = 'desired_name', preview = False)
    ########################
    if preview == True:
    
    table = generateMSQueue(instrument_type = 'Thermo', rowID = ['A','B'], colID = [1,2], rackID = 3, MSmethod = 'desired_method', LCmethod = 'desired_method', instName = 'OA', instNo = 4, evosepNo = 11, 
    gradientID = 'desired_gradient', personalID = 'DeOl', expID = 'experiment', ThermoMethodPath = 'path', number_of_blanks = 2, blankRackID = 2, byRow = False, full = False, add_blanks = True, randomize = True,
    output_path = 'desired_path', output_name = 'desired_name', preview = True)
    """
    
    
    
    
    ###
    cols = [['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']] * 8
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    plate_layout = dict(zip(rows, cols))
    ###
    
    if instrument_type == 'Thermo':
        def getThermoQueue (rowID = rowID, 
                      colID = colID, 
                      rackID = rackID,  
                      method = MSmethod,
                      instName = instName, 
                      instNo = instNo, 
                      evosepNo = evosepNo, 
                      gradientID = gradientID, 
                      personalID = personalID,
                      expID = expID, 
                      ThermoMethodPath = ThermoMethodPath, 
                      full = full, 
                      randomize = randomize,
                      add_blanks = add_blanks,
                      number_of_blanks = number_of_blanks,
                      blankRackID = blankRackID,
                      byRow = byRow):
            def setNameCol (rowID = rowID, colID = colID, instName = instName, instNo = instNo, evosepNo = evosepNo, gradientID = gradientID, personalID = personalID, expID = expID, byRow = byRow):
                dateID= date.today()
                dateID = dateID.strftime("%Y-%m-%d").replace('-','')
                sampleName = []
                wellPosition = []
                
                if byRow is not False:
                    for col in colID:
                        for row in rowID:
                            sampleName.append(dateID+'_'+str(instName)+str(instNo)+'_'+'Evo'+str(evosepNo)+'_'+gradientID+'_'+personalID+'_'+expID+'_'+row+str(col))
                            wellPosition.append(row+str(col))
                else:
                    for row in rowID:
                        for col in colID:
                            sampleName.append(dateID+'_'+str(instName)+str(instNo)+'_'+'Evo'+str(evosepNo)+'_'+gradientID+'_'+personalID+'_'+expID+'_'+row+str(col))
                            wellPosition.append(row+str(col))
                return(sampleName, wellPosition)
    
        ###
            def setEvosepCol (posID, rackID = rackID):
                evosepPos = []
                for pos in posID:
                    evosepPos.append('S'+str(rackID)+':'+str(pos))
                return(evosepPos)
    
        ###
    
            def setBlank (  number = number_of_blanks,
                    rackID = blankRackID,
                    instName = instName, 
                    instNo = instNo, 
                    evosepNo = evosepNo, 
                    gradientID = gradientID, 
                    personalID = personalID, 
                    expID = expID 
                    ):
                dateID= date.today()
                dateID = dateID.strftime("%Y-%m-%d").replace('-','')
                total_size = 2*number
                blank_nums = list(range(0, total_size))
                blank_positions = []
                for key in plate_layout:
                    tmp = plate_layout[key]
                    for el in tmp:
                        blank_positions.append(key+el)
                blankID = []
                blank_wellID = []
                for el in blank_nums:
                    blankID.append(dateID+'_'+str(instName)+str(instNo)+'_'+'Evo'+str(evosepNo)+'_'+gradientID+'_'+personalID+'_'+expID+'_'+'blank'+'_'+str(el))
                    blank_wellID.append('S'+str(rackID)+':'+blank_positions[el])
                return(blankID, blank_wellID)
    
        ###
            if full is not False:
                full_rowID = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                full_colID = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
                filename_col, well_position = setNameCol(rowID=full_rowID, colID= full_colID)
                position_col = setEvosepCol(posID = well_position)
                final_table = pd.DataFrame({'File Name': filename_col,
                                'Path': ['D:\\']*len(filename_col),
                                'Instrument Method':[ThermoMethodPath+method]*len(filename_col),
                                'Position': position_col})
        ###
            if full == False:
                filename_col, well_position = setNameCol()
                position_col = setEvosepCol(posID = well_position)
                final_table = pd.DataFrame({'File Name': filename_col,
                                'Path': ['D:\\']*len(filename_col),
                                'Instrument Method':[ThermoMethodPath+method]*len(filename_col),
                                'Position': position_col})
    
            if randomize is not False: 
                final_table = final_table.sample(frac = 1).reset_index(drop = True)
        ###
            if add_blanks is not False:
                blanksIDs, blanks_positions = setBlank()
                blank_table = pd.DataFrame({'File Name': blanksIDs,
                                    'Path': ['D:\\']*len(blanksIDs),
                                    'Instrument Method':[ThermoMethodPath+method]*len(blanksIDs),
                                    'Position': blanks_positions})
                final_table = pd.concat([blank_table[:number_of_blanks], final_table], ignore_index = True)
                final_table = pd.concat([final_table, blank_table[number_of_blanks:]], ignore_index = True)
            return(final_table)
    ###
     
        output_table = getThermoQueue()
        if preview is not False:
            return(output_table)
        else:
            with open(output_path+output_name, 'w') as file:
                file.write('Bracket Type=4\n')
            output_table.to_csv(output_path+output_name, mode = 'a',index = False)
    
    ###
    ###
    ###
    
    if instrument_type == 'Sciex':
        def getSciexQueue (rowID = rowID,
                   colID = colID,
                   rackID = rackID,
                   MSmethod = MSmethod,
                   LCmethod = LCmethod,
                   instName = instName,
                   instNo = instNo,
                   evosepNo = evosepNo,
                   gradientID = gradientID,
                   personalID = personalID,
                   expID = expID,
                   full = full,
                   randomize = randomize,
                   add_blanks = add_blanks,
                   number_of_blanks = number_of_blanks,
                   blankRackID = blankRackID,
                   byRow = byRow 
                   ):
            def setNameColSciex (rowID = rowID, colID = colID, instName = instName, instNo = instNo, evosepNo = evosepNo, gradientID = gradientID, personalID = personalID, expID = expID, byRow = byRow):
                dateID= date.today()
                dateID = dateID.strftime("%Y-%m-%d").replace('-','')
                sampleName = []
                wellPosition = []
                if byRow is not False:
                    for col in colID:
                        for row in rowID:
                            sampleName.append(dateID+'_'+str(instName)+str(instNo)+'_'+'Evo'+str(evosepNo)+'_'+str(gradientID)+'_'+personalID+'_'+expID+'_'+row+str(col))
                            wellPosition.append(row+col)
                else:
                    for row in rowID:
                        for col in colID:
                            sampleName.append(dateID+'_'+str(instName)+str(instNo)+'_'+'Evo'+str(evosepNo)+'_'+str(gradientID)+'_'+personalID+'_'+expID+'_'+row+str(col))
                            wellPosition.append(row+str(col))
                return(sampleName, wellPosition)
    ###
            def setBlank (  number = number_of_blanks,
                    instName = instName, 
                    instNo = instNo, 
                    evosepNo = evosepNo, 
                    gradientID = gradientID, 
                    personalID = personalID, 
                    expID = expID 
                    ):
                dateID= date.today()
                dateID = dateID.strftime("%Y-%m-%d").replace('-','')
                total_size = 2*number
                blank_nums = list(range(0, total_size))
                blank_positions = []
                for key in plate_layout:
                    tmp = plate_layout[key]
                    for el in tmp:
                        blank_positions.append(key+el)
                blankID = []
                blank_wellID = []
                for el in blank_nums:
                    blankID.append(dateID+'_'+str(instName)+str(instNo)+'_'+'Evo'+str(evosepNo)+'_'+str(gradientID)+'_'+personalID+'_'+expID+'_'+'blank'+'_'+str(el))
                    blank_wellID.append(blank_positions[el])
                return(blankID, blank_wellID)
            if full is not False:
                full_rowID = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                full_colID = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
                filename_col, well_position = setNameColSciex(rowID=full_rowID, colID= full_colID)
                final_table = pd.DataFrame({'Sample Name': filename_col,
                                'MS Method': [MSmethod]*len(filename_col),
                                'LC Method': [LCmethod]*len(filename_col),
                                'Rack Type': ['Evosep One tray']*len(filename_col),
                                'Rack Position':['S'+str(rackID)]*len(filename_col),
                                'Plate Type': ['96 Evotip box']*len(filename_col),
                                'Plate Position': ['Default']*len(filename_col),
                                'Vial Position': well_position,
                                'Data File': filename_col
                                })
            if full == False:
                filename_col, well_position = setNameColSciex()
                final_table = pd.DataFrame({'Sample Name': filename_col,
                                'MS Method': [MSmethod]*len(filename_col),
                                'LC Method': [LCmethod]*len(filename_col),
                                'Rack Type': ['Evosep One tray']*len(filename_col),
                                'Rack Position':['S'+str(rackID)]*len(filename_col),
                                'Plate Type': ['96 Evotip box']*len(filename_col),
                                'Plate Position': ['Default']*len(filename_col),
                                'Vial Position': well_position,
                                'Data File': filename_col
                                })
            
            ###
            
            if randomize is not False: 
                final_table = final_table.sample(frac = 1).reset_index(drop = True)
            if add_blanks is not False:
                blanksIDs, blanks_positions = setBlank()
                blank_table = pd.DataFrame({'Sample Name': blanksIDs,
                                'MS Method': [MSmethod]*len(blanksIDs),
                                'LC Method': [LCmethod]*len(blanksIDs),
                                'Rack Type': ['Evosep One tray']*len(blanksIDs),
                                'Rack Position':['S'+str(blankRackID)]*len(blanksIDs),
                                'Plate Type': ['96 Evotip box']*len(blanksIDs),
                                'Plate Position': ['Default']*len(blanksIDs),
                                'Vial Position': blanks_positions,
                                'Data File': blanksIDs
                                })
                final_table = pd.concat([blank_table[:number_of_blanks], final_table], ignore_index = True)
                final_table = pd.concat([final_table, blank_table[number_of_blanks:]], ignore_index = True)
            return(final_table)
        output_table = getSciexQueue()
        if preview is not False:
            return(output_table)
        else:
            output_table.to_csv(output_path+output_name, mode='a', index = False)
    
    ###
    ###
    ###
         
            