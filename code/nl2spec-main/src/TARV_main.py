import nl2ltl
import ltl2ba
import mitl2ba
import tdlrv
import tarv

import re
import xml.etree.ElementTree as ET

'''
here we have to use:
export PATH=$PATH:~/TDLRV
export PATH=$PATH:~/TA_RV_Predictor/build/src/monitaal-bin
'''

def trace_trans(trace, time_file):
    trace_lis =  [line.decode('utf-8') for line in trace.split(b'\n') if line]
    trace_lis_new = []
    J_now = trace_lis[0]
    alp_now = 'a'
    for J in trace_lis:
        if J != J_now:
            alp_now = chr(ord(alp_now) + 1)
            J_now = J
        trace_lis_new.append(alp_now)


    with open(time_file, 'r') as file1:
        content = file1.readlines()
    
    trace_new_file = '../output/trace.txt'
    lis = []
    with open(trace_new_file, 'w') as file2:
        for i in range(len(trace_lis_new)):
            if trace_lis_new[i] not in lis:
                time_str = '@' + content[i].strip('\n') + ' ' + trace_lis_new[i] + '\n'
                file2.write(time_str)
                lis.append(trace_lis_new[i])
        file2.write('q')

    return trace_new_file

def mitl_trans(mitl):
    
    # ltl = (G(((m & n) U t)), 1.0)
    # ltl_new = ([] p0) || (<> p1)
    
    mitl = str(mitl)
    mitl_new = 'G ('
    for i in range(len(mitl)):
        mitl_new += mitl[i]
    
    mitl_new += ')'
    mitl_new = re.sub('\s+', ' ', mitl_new)
    return mitl_new

def FCQ_MITL(mitl):
    
    new_mitl = mitl

    return new_mitl


def main():
    # use Natural Language trans to LTL
    # bash.sh : python TARV_main.py --model llama_model --keyfile '../input/keyfile/key.txt'  --nl '../input/nl.txt' --num_tries 3 --temperature 0.2 --prompt minimal
    
    mitl = nl2ltl.nl2ltl_main()
    print('='*20 + '1.0' + '='*20)

    new_mitl = FCQ_MITL(mitl)
    
    print('='*20 + '1.0' + '='*20)

    mitl_property = "a -> I_{[0,60]} b"
    mitl_assumption = mitl_property
    # 调整MITL公式, 如添加G( )等
    mitl_property_new = mitl_trans(mitl_property)
    mitl_assumption_new = mitl_trans(mitl_assumption)
    print('-----------------------------FINAL----------------------------')
    print('mitl_property_new: ', mitl_property_new)
    print('mitl_assumption_new: ', mitl_assumption_new)

    # use MITL trans to BA
    # bash.sh : python TARV_main.py -f "([] p0) || (<> p1)" -t -g

    ba_xml = mitl2ba.mitl2ba_main(mitl_property_new, mitl_assumption_new)
    ba_xml_file = '../output/ba.xml'
    print('ba: ', ba_xml)
    xml_tree = ET.ElementTree(ET.fromstring(ba_xml))
    xml_tree.write(ba_xml_file, encoding='utf-8', xml_declaration=True)


    # Raw Message trans to Trace
    # bash.sh : python TARV_main.py
    tdl_file = "../input/TDLRV/test1.tdl"
    raw_file = "../input/TDLRV/rawmessage.txt"
    time_file = "../input/TDLRV/time.txt"
    trace = tdlrv.tdlrv_main(tdl_file, raw_file)
    print('trace: ', trace)

    trace_new_file = trace_trans(trace, time_file)

    # UPPAAL tool
    # bash.sh : ./src/monitaal-bin/MoniTAal-bin -p property ./input/TARV.xml -n assumption ./input/TARV.xml
    
    #ba_xml_file = '../output/ba copy.xml'
    #trace_new_file = '../output/trace copy.txt'
    tarv.tarv_main(ba_xml_file, trace_new_file)
    print('Finished!!!')


if __name__ == "__main__":
    main()

