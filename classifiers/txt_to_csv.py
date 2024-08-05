import os
import pandas as pd
from get_number import get_last_number_pos

#*===============================================================================*#
#Setup:
#   You need to create 2 different folders, one for input and another for output
#   In the input folder, there may only be .txt files
#   It is necessary to install the following packages and all their dependencies:
#       pandas
#       openpyxl (if Excel format will be used)
#*===============================================================================*#
is_csv = 1 #Sets the format that the output will be saved as (0 is for Excel)
input_folder = "input" #Sets which folder the .txt will be read from
output_folder = "output" #Sets which folder the output file will be saved on
#*===============================================================================*#


for file in os.listdir(input_folder):
    path = os.path.join(input_folder, file)
    with open(path) as f:
        lines = f.readlines()

    first_block = 1
    df_null = pd.DataFrame({"A":[""]})

    array_first = []
    array = []
    col = 1
    first = 1
    for i in range(len(lines)):
        if (lines[i][:11] == "           "):
            if (first):
                place = lines[i-1].rfind(">")
                array_first.append(lines[i-1][place+1:-1]) #Coutry and city

            end = lines[i].find(" ", 11)

            name = lines[i][11:end]
            array.append(name) #Score name

            mid = lines[i].find("=", end)
            while (mid != -1):
                dname = lines[i][end+1:mid]

                if (lines[i][mid+1] == "("): #Interval case
                    start_num = mid+2
                    j = 0
                    while (lines[i][start_num] != ")"):
                        end = get_last_number_pos(lines[i], start_num) + 1
                        if (end > 0):
                            if (first):
                                array_first.append(dname + f"_{j}") #Name of intervals
                            j+=1
                            dnumber = float(lines[i][start_num:end])
                            array.append(dnumber) #Value of interval data
                            start_num = end
                        else:
                            start_num += 1
                    end = start_num + 1
                        
                else:
                    if (first):
                        array_first.append(dname) #Name of the type of the analysis
                        
                    end = get_last_number_pos(lines[i], mid+1) + 1

                    dnumber = float(lines[i][mid+1:end])
                    array.append(dnumber) #Value of data

                mid = lines[i].find("=", end)
            
            if(first):
                first = 0
                df_temp = pd.DataFrame({"A": array_first, "B": array})
                array_first.clear()
            else:
                letter = chr(ord("A") + col)
                df_col = pd.DataFrame({letter:array})
                df_temp = df_temp.join(df_col)
            array.clear()
            col += 1
            if (lines[i+1][:11] != "           "):
                if (first_block):
                    df = df_temp
                    first_block = 0
                else:
                    df = pd.concat([df, df_null], ignore_index=True)
                    df = pd.concat([df, df_temp], ignore_index=True)

                first = 1
                col = 1

    path = os.path.join(output_folder, file[:-3])
    if(is_csv):
        df.to_csv(path + "csv", index=False, header=False)
    else:
        df.to_excel(path + "xlsx", sheet_name="sheet teste", index=False, header=False)


    