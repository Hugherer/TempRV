import re
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

def extract_time_numbers(s):
    # 定义正则表达式，匹配形如 "{[0,5]}" 的字符串并捕获数字
    pattern = r'\{\[(\d+),\s*(\d+)\]\}'
    
    # 使用 re.search 搜索匹配项
    match = re.search(pattern, s)
    
    if match:
        # 提取捕获的数字
        num1 = int(match.group(1))  # 第一个数字
        num2 = int(match.group(2))  # 第二个数字
        return num1, num2
    else:
        # 如果字符串不符合格式，返回 None 或抛出异常
        print("don't match!!!")
        return None


class TimedAutomaton:
    def __init__(self, num = 0):
        self.states = []
        self.transitions = []
        self.current_state_id = num

    def add_state(self, state_name):
        state_id = f"id{self.current_state_id}"
        self.current_state_id += 1
        self.states.append((state_id, state_name))
        return state_id

    def add_transition(self, from_state, to_state, action, time_constraint=None):
        transition = {
            'from': from_state,
            'to': to_state,
            'action': action,
            'time_constraint': time_constraint
        }
        self.transitions.append(transition)

    def initialize_from_mitl(self, mitl_formula):
        # 解析MITL公式
        tokens = self.tokenize(mitl_formula)
        for i in tokens:
            if i == '' or len(tokens) < 3: 
                print('tokenize must be something wrong, we have to add new Regular Expression!')
                break
        print('tokens: ', tokens)
        id_num = 0
        start_state = self.add_state(f'q{id_num}')
        current_state = start_state
        stack = []
        for i, token in enumerate(tokens):
            if token <= 'Z' and token >= 'A':         # A~Z
                if token == 'G':
                    continue

            elif token <= 'z' and token >= 'a':       # a~z
                id_num += 1
                next_state = self.add_state(f'q{id_num}')
                action = token
                if re.match(r'^\{\[\d+,\s*\d+\]\}$', tokens[i - 1]):      # match "{[0, 5]}"
                    interval = tokens[i - 1]
                else:
                    interval = None
                self.add_transition(current_state, next_state, action, interval)
                current_state = next_state
            else:
                continue
        '''
        for i, token in enumerate(tokens):
            if token == '->':
                continue
            elif token.startswith(('X_', 'U_', 'I_', 'G_')):
                interval = self.extract_interval(tokens[i + 1])         # 
                signal = tokens[i + 1]
                next_state = self.add_state(f'q{len(self.states)}')

                if token.startswith('X_'):
                    self.add_transition(current_state, next_state, f"{signal} == True", interval)
                elif token.startswith('U_'):
                    end_signal = tokens[i + 2]
                    self.add_transition(current_state, next_state, f"{signal} == True and {end_signal} == False", interval)
                    self.add_transition(next_state, next_state, f"{signal} == True and {end_signal} == False", interval)
                    final_state = self.add_state(f'q{len(self.states)}')
                    self.add_transition(next_state, final_state, f"{end_signal} == True")
                    current_state = final_state
                elif token.startswith('I_'):
                    self.add_transition(current_state, next_state, f"{signal} == True", interval)
                elif token.startswith('G_'):
                    self.add_transition(current_state, next_state, f"{signal} == True", interval)
                    current_state = next_state
            else:
                stack.append(token)
        '''
        return id_num


    def tokenize(self, formula):
        # 修改后的正则表达式
        pattern = r'\b\w+\b|\->|\{[^}]*\}|\[[^\]]*\]|\(|\)|\s+'
        return [token for token in re.findall(pattern, formula) if token.strip()]

    def extract_interval(self, token):
        match = re.search(r'\{(\d+),(\d+)\}', token)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None

    def print_timed_automaton_xml(self, bool):
        if bool == 'property':
            template = Element('template')
            name = SubElement(template, 'name')
            name.text = 'property'

            init_ref = None
            state_i = 0
            for state_id, state_name in self.states:
                if state_i == 0: init_ref = state_id
                state_i += 1
                location = SubElement(template, 'location', id=state_id)
                location_name = SubElement(location, 'name')
                if state_i == len(self.states):
                    location_name.text = state_name + '_a'
                else:
                    location_name.text = state_name
                
            if init_ref:
                init = SubElement(template, 'init', ref=init_ref)

            trans_len = len(self.transitions)
            trans_len_id = 0
            for transition in self.transitions:
                trans = SubElement(template, 'transition')
                source = SubElement(trans, 'source', ref=transition['from'])
                target = SubElement(trans, 'target', ref=transition['to'])

                if transition['time_constraint']:
                    sync = SubElement(trans, 'label', kind='synchronisation')
                    sync.text = f"{transition['action']}!"

                    guard = SubElement(trans, 'label', kind='guard')
                    min_time, max_time = extract_time_numbers(transition['time_constraint'])
                    guard.text = f"x >= {min_time} && x <= {max_time}"
        
                else :
                    sync = SubElement(trans, 'label', kind='synchronisation')
                    sync.text = f"{transition['action']}!"

                    assignment = SubElement(trans, 'label', kind='assignment')
                    assignment.text = "x := 0"

                trans_len_id += 1

                if trans_len_id == trans_len:
                    trans = SubElement(template, 'transition')
                    source = SubElement(trans, 'source', ref=transition['to'])
                    target = SubElement(trans, 'target', ref=transition['to'])
                    sync = SubElement(trans, 'label', kind='synchronisation')
                    sync.text = f"{'a'}!"

                    
            rough_string = tostring(template, 'utf-8').decode('utf-8')
            #reparsed = parseString(rough_string)
            #pretty_xml = reparsed.toprettyxml(indent="  ")
            #print(pretty_xml)
            return rough_string
        elif bool == 'assumption':
            template = Element('template')
            name = SubElement(template, 'name')
            name.text = 'assumption'

            init_ref = None
            state_i = 0
            for state_id, state_name in self.states:
                if state_i == 0: init_ref = state_id
                state_i += 1
                location = SubElement(template, 'location', id=state_id)
                location_name = SubElement(location, 'name')
                if state_i == len(self.states):
                    location_name.text = state_name + '_a'
                else:
                    location_name.text = state_name
                

            if init_ref:
                init = SubElement(template, 'init', ref=init_ref)

            trans_len = len(self.transitions)
            trans_len_id = 0
            for transition in self.transitions:
                trans = SubElement(template, 'transition')
                source = SubElement(trans, 'source', ref=transition['from'])
                target = SubElement(trans, 'target', ref=transition['to'])

                if transition['time_constraint']:
                    sync = SubElement(trans, 'label', kind='synchronisation')
                    sync.text = f"{transition['action']}!"

                    guard = SubElement(trans, 'label', kind='guard')
                    min_time, max_time = extract_time_numbers(transition['time_constraint'])
                    guard.text = f"x >= {min_time} && x <= {max_time}"
        
                else :
                    sync = SubElement(trans, 'label', kind='synchronisation')
                    sync.text = f"{transition['action']}!"

                    assignment = SubElement(trans, 'label', kind='assignment')
                    assignment.text = "x := 0"

                trans_len_id += 1

                if trans_len_id == trans_len:
                    trans = SubElement(template, 'transition')
                    source = SubElement(trans, 'source', ref=transition['to'])
                    target = SubElement(trans, 'target', ref=transition['to'])
                    sync = SubElement(trans, 'label', kind='synchronisation')
                    sync.text = f"{'a'}!"

            rough_string = tostring(template, 'utf-8').decode('utf-8')
            #reparsed = parseString(rough_string)
            #pretty_xml = reparsed.toprettyxml(indent="  ")
            return rough_string
        else: print('here must be something wrong!!')

def mitl2ba_main(mitl_property_new, mitl_assumption_new):
    ta_1 = TimedAutomaton()

    num = ta_1.initialize_from_mitl(mitl_property_new)
    ba_xml_property = ta_1.print_timed_automaton_xml('property')

    ta_2 = TimedAutomaton(num + 1)
    ta_2.initialize_from_mitl(mitl_assumption_new)
    ba_xml_assumption = ta_2.print_timed_automaton_xml('assumption')

    title = '<declaration> \n broadcast chan a, b, c, d, m; \n clock x;</declaration>'
    ba_xml = '<nta>' + title + ba_xml_property + ba_xml_assumption + '</nta>'
    reparsed = parseString(ba_xml)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    return pretty_xml