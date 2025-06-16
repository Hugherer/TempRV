'''
this is added by hugh !!!
which is used for UPPAAL TOOL
'''


from subprocess import Popen, PIPE
import re
import argparse
import sys
import __main__


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--formula",
                       help="translate LTL into never claim", type=str)
    group.add_argument("-F", "--file",
                       help="like -f, but with the LTL formula stored in a "
                       "1-line file", type=argparse.FileType('r'))
    parser.add_argument("-d",
                        help="display automata (D)escription at each step",
                        action='store_true')
    parser.add_argument("-s",
                        help="computing time and automata sizes (S)tatistics",
                        action='store_true')
    parser.add_argument("-l",
                        help="disable (L)ogic formula simplification",
                        action='store_true')
    parser.add_argument("-p",
                        help="disable a-(P)osteriori simplification",
                        action='store_true')
    parser.add_argument("-o",
                        help="disable (O)n-the-fly simplification",
                        action='store_true')
    parser.add_argument("-c",
                        help="disable strongly (C)onnected components "
                        "simplification", action='store_true')
    parser.add_argument("-a",
                        help="disable trick in (A)ccepting conditions",
                        action='store_true')
    parser.add_argument("-g", "--graph",
                        help="display buchi automaton graph",
                        action='store_true')
    parser.add_argument("-G", "--output-graph",
                        help="save buchi automaton graph in pdf file",
                        type=argparse.FileType('w'))
    parser.add_argument("-t", "--dot",
                        help="print buchi automaton graph in DOT notation",
                        action='store_true')
    parser.add_argument("-T", "--output-dot",
                        help="save buchi automaton graph in DOT file",
                        type=argparse.FileType('w'))
    return parser.parse_args()


def get_ltl_formula(file, formula):
    assert file is not None or formula is not None
    if file:
        try:
            ltl = file.read()
        except Exception as e:
            eprint("{}: {}".format(__main__.__file__, str(e)))
            sys.exit(1)
    else:
        ltl = formula
    ltl = re.sub('\s+', ' ', ltl)
    if len(ltl) == 0 or ltl == ' ':
        eprint("{}: empty ltl formula.".format(__main__.__file__))
        sys.exit(1)
    return ltl


def run_tarv(ba_xml, trace_new):
    tarv_args = ["MoniTAal-bin", "-p", "property", ba_xml, "-n", "assumption", ba_xml, "-i", trace_new]

    try:
        process = Popen(tarv_args, stdout=PIPE)
        (output, err) = process.communicate()
        exit_code = process.wait()
    except FileNotFoundError as e:
        eprint("{}: parsertest not found.\n".format(__main__.__file__))
        eprint("compile the sources and add the binary to your $PATH, e.g.\n")
        eprint("\t~$ export PATH=$PATH:path-to-MoniTAal-bin-dir\n")
        sys.exit(1)

    # output = output.decode('utf-8')

    return output, err, exit_code

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def tarv_main(ba_xml, trace_new):
    print('........1.1')
    output = run_tarv(ba_xml, trace_new)
    
    print(output)
    
    return 0