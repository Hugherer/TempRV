from argparse import Namespace
from flask import Flask, render_template, request, url_for, g
import backend
import pandas as pd
import os
import json
from time import time

app = Flask(__name__, template_folder="../templates", static_folder="../static")



@app.before_request
def before_request():
    g.start_time = time()

@app.after_request
def after_request(response):
    if hasattr(g, 'start_time'):
        elapsed_time = time() - g.start_time
        print(f"Request {request.path} took {elapsed_time:.4f} seconds.")
        # 如果你想在响应头中包含这个信息
        response.headers['X-Processing-Time'] = f"{elapsed_time:.4f} seconds"
    return response

@app.route("/", methods=["POST", "GET"])
@app.route("/home/", methods=["POST", "GET"])
@app.route("/nl2ltl/", methods=["POST", "GET"])
def home():
    form_data = request.form
    form_data = request.form
    if request.method == "POST":
        subtranslation_map, locked_subtranslation_map = subtranslation_gen(form_data)
        prompt = form_data["prompts"]

        FCQs_prompt = form_data["FCQs_prompts"]

        input = form_data["nl"]
        num_tries = int(form_data["num_tries"])
        temperature = float(form_data["temperature"]) * 0.1
        ground_truth = form_data["ground_truth"] if "ground_truth" in form_data else ""
        keyfile = ""
        if (
            form_data["models"] == "code-davinci-002"
            or form_data["models"] == "text-davinci-003"
            or form_data["models"] == "gpt-3.5-turbo"
            or form_data["models"] == "code-davinci-edit-001"
            or form_data["models"] == "gpt-4"
        ):
            keyfile = os.path.join("..", "keys", "oai_key.txt")
        if form_data["models"] == "bloom" or form_data["models"] == "bloomz":
            keyfile = os.path.join("..", "keys", "hf_key.txt")
        if form_data["models"] == "text-bison@001" or form_data["models"] == "code-bison@001":
            keyfile = os.path.join("..", "keys", "google_project_id.txt")
        if form_data["models"] == "deepseek_model" or form_data["models"] == "llama_model" or form_data["models"] == "qwen_model" \
            or form_data["models"] == "llama3b_model" or form_data["models"] == "qwen7b_model":
            keyfile = os.path.join("..", "input/keyfile", "key.txt")
        
        print("-"*20 + '3.0' + '-'*20)
        ns = Namespace(
            keyfile=keyfile,
            maxtokens=128,
            model=form_data["models"],
            nl=input,
            prompt=prompt,
            given_translations=str(subtranslation_map),
            locked_translations=str(locked_subtranslation_map),
            num_tries=num_tries,
            temperature=temperature,
        )
        res = backend.call(ns)
        final_formula = res[0][0]
        certainty = res[0][1]
        
        ns_new = Namespace(
            keyfile=keyfile,
            maxtokens=128,
            model=form_data["models"],
            nl=input,
            prompt=FCQs_prompt,
            mitl=res[0][0],
            given_translations=str(subtranslation_map),
            locked_translations=str(locked_subtranslation_map),
            num_tries=num_tries,
            temperature=temperature,
        )

        FCQs_res = backend.FCQs_call(ns_new)
        print("-"*20 + 'FCQs_res...1' + '-'*20)
        print(FCQs_res)
        print("-"*20 + 'FCQs_res...2' + '-'*20)
        print(FCQs_res[0])
        print("-"*20 + 'FCQs_res...3' + '-'*20)
        print(FCQs_res[1])
        print("-"*20 + 'FCQs_res...4' + '-'*20)
        print(FCQs_res[2])
        print("-"*20 + 'FCQs_res...4' + '-'*20)

        # here
        subtranslations = [FCQs_res[0], FCQs_res[1],FCQs_res[2]]
        tmp = []
        for i in range(len(FCQs_res[2])):
            tmp.append('0')
        subtranslations.append(tmp)

        #print("-"*20 + 'subtranslations' + '-'*20)
        #print(subtranslations)
        #print("-"*20 + 'subtranslations' + '-'*20)
        #subtranslations = [['qwe', 'qwere'],['sdf', 'serfs'],['yes','no'],['0','0']]


        return render_template(
            "home.html",
            examples=load_examples(),
            ground_truth=ground_truth,
            final_output=final_formula,
            certainty=str(round(certainty * 100, 2)) + "%",
            input=input,
            subnl="",
            subltl="",
            num_tries=num_tries,
            subtranslations=subtranslations,
            models=form_data["models"],
            prompts=prompt,
            temperature=form_data["temperature"],
        )
    return render_template(
        "home.html",
        examples=load_examples(),
        ground_truth="",
        num_tries=1,
        models="llama",
        prompts="MITL",
        temperature=2,
        input="../input/mitl.txt",
    )


def load_examples():
    return json.dumps(
        pd.read_csv(os.path.join("..", "examples.csv"), delimiter=";").values.tolist()
    )


def subtranslation_gen(form_data):
    # number of fixed inputs: nl, prompt, model, temperature, runs = 5
    print("-"*20 + '3.1' + '-'*20)
    print(form_data)
    sub_nl = []
    sub_ltl = []
    sub_lock = []
    sub_ids = [
        int(a[9:])
        for a in filter(
            lambda a: a.startswith("examplenl"), [key for key in form_data.keys()]
        )
    ]

    for i in sub_ids:
        nl = form_data["examplenl" + str(i)]
        ltl = form_data["exampleltl" + str(i)]
        locked = "lock" + str(i) in form_data
        if nl != "" and ltl != "":
            sub_nl.append(nl)
            sub_ltl.append(ltl)
            sub_lock.append(locked)
    subtranslation_map = {}
    locked_subtranslation_map = {}
    if sub_nl:
        for i in range(0, len(sub_nl)):
            subtranslation_map[sub_nl[i]] = sub_ltl[i]
            if sub_lock[i]:
                locked_subtranslation_map[sub_nl[i]] = sub_ltl[i]
    return subtranslation_map, locked_subtranslation_map
