import os
import ambiguity
from ltlf2dfa.parser.ltlf import LTLfParser
import ast
import re


def parse_formulas(choices):
    parser = LTLfParser()
    parsed_result_formulas = []
    for c in choices:
        try:
            formula_str = c.split("FINAL:")[1].split("\n")[0].strip(".")
            print('.................1.0')
        except:
            formula_str = ''
            # continue
        
        if formula_str == '':
            try:
                formula_str_list = c.split("could be:")[1].split("\n")
                for c in formula_str_list:
                    if len(c) > 3:
                        formula_str = c
                        break
                formula_str = formula_str.strip('.')
                formula_str = formula_str.strip('\`')
            except:
                formula_str = ''
        if formula_str == '':
            try:
                formula_str_list = c.split("would be:")[1].split("\n")
                for c in formula_str_list:
                    if len(c) > 3:
                        formula_str = c
                        break
                formula_str = formula_str.strip('.')
                formula_str = formula_str.strip('\'')
            except:
                formula_str = ''

        if formula_str != '':
            try:
                parsed_formula = parser(formula_str)
                parsed_result_formulas.append(parsed_formula)
            except:
                parsed_result_formulas.append(formula_str)
        else:
            print("please run again!")
    return parsed_result_formulas


def parse_explanation_dictionary(choices, nl):
    parsed_explanation_results = []
    for c in choices:
        try:
            dict_string = (
                "{" + c.split("dictionary")[1].split("{")[1].split("}")[0] + "}"
            )
            parsed_dict = ast.literal_eval(dict_string)
            parsed_dict = dict(filter(lambda x: x[0] != nl, parsed_dict.items()))
            if parsed_dict:
                parsed_explanation_results.append(parsed_dict)
        except:
            pass
    return parsed_explanation_results


def generate_intermediate_output(intermediate_translation):
    nl = []
    ltl = []
    cert = []
    locked = []
    for t in intermediate_translation:
        nl.append(t[0])
        ltl.append(t[1])
        cert.append(t[2])
        locked.append(t[3])
    return [nl, ltl, cert, locked]


def prompt(args):
    nl_filename = args.nl
    nl_input = ""
    with open(nl_filename, 'r') as file:
        nl_input = file.readline().strip("\n")
    inpt = args.nl
    prompt_dir = os.path.join("..", "input/prompts")
    if args.prompt == "minimal":
        fixed_prompt_file = open(os.path.join(prompt_dir, "minimal.txt"))
        fixed_prompt = fixed_prompt_file.read()
    elif args.prompt == "minimal_par":
        fixed_prompt_file = open(os.path.join(prompt_dir, "minimal_par.txt"))
        fixed_prompt = fixed_prompt_file.read()
    elif args.prompt == "smart":
        fixed_prompt_file = open(os.path.join(prompt_dir, "smart.txt"))
        fixed_prompt = fixed_prompt_file.read()
    elif args.prompt == "tdl":
        fixed_prompt_file = open(os.path.join(prompt_dir, "tdl.txt"))
        fixed_prompt = fixed_prompt_file.read()
    elif args.prompt == "MITL":
        fixed_prompt_file = open(os.path.join(prompt_dir, "MITL.txt"))
        fixed_prompt = fixed_prompt_file.read()
    elif args.prompt == "stl":
        fixed_prompt_file = open(os.path.join(prompt_dir, "stl.txt"))
        fixed_prompt = fixed_prompt_file.read()
    elif args.prompt == "indistribution":
        fixed_prompt_file = open(os.path.join(prompt_dir, "indistribution.txt"))
        fixed_prompt = fixed_prompt_file.read()
    elif args.prompt == "amba_master":
        fixed_prompt_file = open(
            os.path.join(prompt_dir, "amba_master_assumptions.txt")
        )
        fixed_prompt = fixed_prompt_file.read()
    elif args.prompt == "amba_slave":
        fixed_prompt_file = open(os.path.join(prompt_dir, "amba_slave_guarantees.txt"))
        fixed_prompt = fixed_prompt_file.read()
    else:
        fixed_prompt = args.prompt
    if args.given_translations != "":
        final_prompt = (
            fixed_prompt
            + "\nNatural Language: "
            + nl_input
            + "\nGiven translations: "
            + args.given_translations
            + "\nExplanation:"
        )
    else:
        final_prompt = (
            fixed_prompt
            + "\nNatural Language: "
            + nl_input
            + "\nGiven translations: {}"
            + "\nExplanation:"
        )
    #print("FINAL PROMPT:")
    #print(final_prompt)
    return final_prompt


def extract_subinfo(choices, args, n):
    parsed_result_formulas = parse_formulas(choices)

    print("Results of multiple runs:")
    print(parsed_result_formulas)
    final_translation = ambiguity.ambiguity_final_translation(parsed_result_formulas, n)
    parse_explain = parse_explanation_dictionary(choices, args.nl)
    intermediate_translation = ambiguity.ambiguity_detection_translations(
        parse_explain,
        n,
        ast.literal_eval(args.locked_translations)
        if "locked_translations" in args
        else {},
    )
    intermediate_output = generate_intermediate_output(intermediate_translation)
    return final_translation, intermediate_output


def FCQs_prompt(args):
    nl_filename = args.nl
    nl_input = ""
    with open(nl_filename, 'r') as file:
        nl_input = file.readline().strip("\n")
    inpt = args.nl
    prompt_dir = os.path.join("..", "input/prompts")
    if args.prompt == "minimal":
        fixed_prompt_file = open(os.path.join(prompt_dir, "minimal.txt"))
        fixed_prompt = fixed_prompt_file.read()
    elif args.prompt == "amba_master":
        fixed_prompt_file = open(
            os.path.join(prompt_dir, "amba_master_assumptions.txt")
        )
        fixed_prompt = fixed_prompt_file.read()
    elif args.prompt == "amba_slave":
        fixed_prompt_file = open(os.path.join(prompt_dir, "amba_slave_guarantees.txt"))
        fixed_prompt = fixed_prompt_file.read()
    else:
        fixed_prompt = args.prompt
    '''
    if args.given_translations != "":
        final_prompt = (
            fixed_prompt
            + "\nNatural Language: "
            + nl_input
            + "\nGiven translations: "
            + args.given_translations
            + "\nExplanation:"
        )
    else:
        final_prompt = (
            fixed_prompt
            + "\nNatural Language: "
            + nl_input
            + "\nGiven translations: {}"
            + "\nExplanation:"
        )
    '''
    final_prompt = (
        fixed_prompt
        + "\nNatural Language: "
        + nl_input
        + "\nMITL formula: "
        + args.mitl
    )
    #print("FINAL PROMPT:")
    #print(final_prompt)
    return final_prompt

def FCQs_extract_subinfo(FCQ_output, args):
    
    # 正则表达式匹配每一个 FCQ 块
    pattern = r'FCQ \d+:(.*?)\nMITL formula:(.*?)\n(?:Natural Language:.*?\n)?Answer:\s*(Yes|No)'
    matches = re.findall(pattern, FCQ_output, re.DOTALL)

    fcqs = []
    mitl_formulas = []
    answers = []

    for match in matches:
        question = match[0].strip()
        formula = match[1].strip()
        answer = match[2].strip().lower()

        fcqs.append(question)
        mitl_formulas.append(formula)
        answers.append(answer)

    return fcqs, mitl_formulas, answers