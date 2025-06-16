import random
import glob

def merge_txt_files(input_folder, output_file, number):
    if not input_folder.endswith('/'):
        input_folder += '/'
    
    files = glob.glob(f"{input_folder}*.txt")
    
    files.sort()
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        num = 0
        file_num = 0
        for file in files:
            file_num += 1
            with open(file, 'r', encoding='utf-8') as infile:
                if file_num < 7:
                    p = random.randint(1, 20)
                else:
                    p = number - num

                tmp = infile.read()
                for i in range(p):
                    outfile.write(tmp)
                    num += 1
                if num >= number:
                    break
            

def time_line(time_folder, number):
    with open(time_folder, 'w', encoding='utf-8') as outfile:
        for i in range(number):
            outfile.write(str(i+1))
            outfile.write('\n')

if __name__ == "__main__":
    number = 100
    input_folder = '../input/rawmessage/'
    output_file = '../input/TDLRV/rawmessage.txt'
    merge_txt_files(input_folder, output_file, number)

    time_folder = '../input/TDLRV/time.txt'
    #time_line(time_folder, number)