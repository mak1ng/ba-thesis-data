import csv
import pathlib
import re
import subprocess
import pygments
from pygments.lexers import PythonLexer
from pygments.token import Comment, Literal
import libcst as cst
import os
from subprocess import PIPE
import numpy as np
import scipy.sparse
import cntk as C
import argparse

#get arguments
parser = argparse.ArgumentParser(description='Creates annotated versions of a given module under use of Deeptyper')
parser.add_argument('-m','--module', help='Path of the module to be annotated', required=True)
parser.add_argument('-n','--number-of-preds', help='Number of predictions', required=True, type=int)
args = vars(parser.parse_args())

#define inputs and outputs
inp = args["module"]
outp_name = os.path.splitext(inp)[0]
source_file = "tokens.vocab"
target_file = "types.vocab"
model_file = "model-5.cntk"
number_of_predictions = args["number_of_preds"]
results_invalid = str(pathlib.Path(__file__).parent.resolve())+"/results_invalid.csv"

# load dictionaries
source_wl = [line.rstrip('\n') for line in open(source_file)]
target_wl = [line.rstrip('\n') for line in open(target_file)]
source_dict = {source_wl[i]:i for i in range(len(source_wl))}
target_dict = {target_wl[i]:i for i in range(len(target_wl))}

# number of words in vocab, slot labels, and intent labels
vocab_size = len(source_dict)
num_labels = len(target_dict)
epoch_size = 15073245
minibatch_size = 5000
emb_dim = 300
hidden_dim = 650
num_epochs = 10

x = C.sequence.input_variable(vocab_size, name="x")
y = C.sequence.input_variable(num_labels, name="y")
t = C.sequence.input_variable(hidden_dim, name="t")

def BiRecurrence(fwd, bwd):
    F = C.layers.Recurrence(fwd)
    G = C.layers.Recurrence(bwd, go_backwards=True)
    x = C.placeholder()
    apply_x = C.splice(F(x), G(x))
    return apply_x

def create_model():
    embed = C.layers.Embedding(emb_dim, name='embed')
    encoder = BiRecurrence(C.layers.GRU(hidden_dim//2), C.layers.GRU(hidden_dim//2))
    recoder = BiRecurrence(C.layers.GRU(hidden_dim//2), C.layers.GRU(hidden_dim//2))
    project = C.layers.Dense(num_labels, name='classify')
    do = C.layers.Dropout(0.5)
    
    def recode(x, t):
        inp = embed(x)
        inp = C.layers.LayerNormalization()(inp)
        
        enc = encoder(inp)
        rec = recoder(enc + t)
        proj = project(do(rec))
        
        dec = C.ops.softmax(proj)
        return enc, dec
    return recode

def criterion(model, labels):
    ce	 = -C.reduce_sum(labels*C.ops.log(model))
    errs = C.classification_error(model, labels)
    return ce, errs

def enhance_data(data, enc):
    guesses = enc.eval({x: data[x]})
    inputs = C.ops.argmax(x).eval({x: data[x]})
    tables = []
    for i in range(len(inputs)):
        ts = []
        table = {}
        counts = {}
        for j in range(len(inputs[i])):
            inp = int(inputs[i][j])
            if inp not in table:
                table[inp] = guesses[i][j]
                counts[inp] = 1
            else:
                table[inp] += guesses[i][j]
                counts[inp] += 1
        for inp in table:
            table[inp] /= counts[inp]
        for j in range(len(inputs[i])):
            inp = int(inputs[i][j])
            ts.append(table[inp])
        tables.append(np.array(np.float32(ts)))
    s = C.io.MinibatchSourceFromData(dict(t=(tables, C.layers.typing.Sequence[C.layers.typing.tensor])))
    mems = s.next_minibatch(minibatch_size)
    data[t] = mems[s.streams['t']]

def create_trainer():
    masked_dec = dec*C.ops.clip(C.ops.argmax(y), 0, 1)
    loss, label_error = criterion(masked_dec, y)
    loss *= C.ops.clip(C.ops.argmax(y), 0, 1)

    lr_schedule = C.learning_parameter_schedule_per_sample([1e-4]*2 + [5e-5]*2 + [1e-6], epoch_size=int(epoch_size))
    momentum_as_time_constant = C.momentum_as_time_constant_schedule(1000)
    learner = C.adam(parameters=dec.parameters,
                         lr=lr_schedule,
                         momentum=momentum_as_time_constant,
                         gradient_clipping_threshold_per_sample=15, 
                         gradient_clipping_with_truncation=True)

    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=num_epochs)
    trainer = C.Trainer(dec, (loss, label_error), learner, progress_printer)
    trainer.restore_from_checkpoint(model_file)
    C.logging.log_number_of_parameters(dec)
    return trainer

def has_return(node: cst.CSTNode)->bool or None:
    """Recursively yield all descendant nodes of the given node and search for possible return statement"""
    for child in node.children:
        if isinstance(child, cst.Return):
            return True
        elif child.children is not None:
            has_re = has_return(child)
            if has_re is True:
                return has_re

class AnnotationTransformer(cst.CSTTransformer):
    """Removes all annotations from functions, function parameters and assigns"""

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:
        return updated_node.with_changes(returns=None)
    
    def leave_Param(self, original_node: cst.Param, updated_node: cst.Param) -> cst.CSTNode:
        return updated_node.with_changes(annotation=None)

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.CSTNode:
        return updated_node
    
    def leave_AnnAssign(self, original_node: cst.AnnAssign, updated_node: cst.Assign) -> cst.CSTNode:
        if original_node.value != None:
            return cst.Assign([cst.AssignTarget(original_node.target)], original_node.value)
        else:
            return updated_node

class MarkingTransformer(cst.CSTTransformer):
    """Marks all functions, function parameters and assigns that can potentially be annotated with a type. Marking is done by adding
    ___MF for functions, ___MP for params and ___MA for assigns at the beginning and the end of their name."""

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:
        if original_node.returns is None:
            #function is only annotable if it has at least one instance of cst.Return in their body
            if has_return(original_node):
                return updated_node.with_changes(name=cst.Name(value="___MF" + original_node.name.value + "FM___",lpar=[],rpar=[],))
            else:
                return updated_node
        else:
            return updated_node

    def leave_Param(self, original_node: cst.Param, updated_node: cst.Param) -> cst.CSTNode:
        if original_node.annotation is None:
            return updated_node.with_changes(name=cst.Name(value="___MP" + original_node.name.value + "PM___", lpar=[],rpar=[]))
        else:
            return updated_node

    def leave_Lambda(self, original_node: cst.Lambda, updated_node: cst.Lambda) -> cst.CSTNode:
        #lambda parameters cannot be annotated
        for c in original_node.children:
            if isinstance(c, cst.Parameters):
                for child in c.children:
                    if isinstance(child, cst.Param):
                        child.with_changes(annotation=cst.Annotation(cst.Name(value="___LAMBDA", lpar=[],rpar=[])))
        return original_node

   

#TODO annotated Vars without value, not removing their annotation because of syntax
def create_tokens(file):
    """converts the given file into tokens used to feed the model and create the outputfiles"""

    #parse the input file into concrete syntax tree
    tree = cst.parse_module(file.read())
    anns_transformer = AnnotationTransformer()
    marked_transformer = MarkingTransformer()
    
    #cst without annotations
    cleaned_tree = tree.visit(anns_transformer)
    #create a file without any annotations
    with open(outp_name + '_clean.py', 'w', encoding="utf-8", newline="") as clean_file:
        clean_file.write(cleaned_tree.code)
    #cst including marks for annotable nodes
    marked_tree = tree.visit(marked_transformer)
    #cst without original anns but with marks for possible anns
    cleaned_marked_tree = cleaned_tree.visit(marked_transformer)

    #use Lexer to convert trees into tokens
    tokens_cleaned  = list(pygments.lex(cleaned_tree.code, PythonLexer()))
    tokens_cleaned_marked = list(pygments.lex(cleaned_marked_tree.code, PythonLexer()))
    tokens_marked = list(pygments.lex(marked_tree.code, PythonLexer()))
    model_toks = []

    #prepare tokens to fit the trainingsdata format
    for t, v in tokens_cleaned:
        tok = ""
        #join some tokens because...
        #...lexer splits operators and isequal sign
        if len(model_toks) > 0 and model_toks[-1] in ["+", "-", ">", "<"] and v == "=":
            model_toks[-1] = model_toks[-1] + v
        #...lexer splits ellipsis in single tokens
        elif len(model_toks) > 1 and model_toks[-2] == "." and model_toks[-1] in "." and v == ".":
            model_toks[-2] = "..."
            model_toks.pop()
        #...lexer splits quotationmarks in single tokens
        elif len(model_toks) > 1 and (model_toks[-2], v) in [("\"", "\""), ("\'", "\'")] and model_toks[-1] == "[string]":
            model_toks[-2] = "[string]"
            model_toks.pop()
        elif v.strip() == "" and v != "\n":
            continue
        elif t in Literal.String.Doc:
            tok = "[docstring] "
        elif t in Literal.Number:
            tok = "[number]"
        elif v == "\n":
            tok = "[EOL]"
        elif t in Comment:
            tok = "[comment]"
        elif v in ["\"", "\'"]:
            model_toks.append(v)
        elif t in Literal.String :
            tok = "[string]"
        else:
            if v in source_dict:
                tok = v
            else:
                tok = "_UNKNOWN_"
        if tok != "":
            model_toks.append(tok)
            
    marked_toks = []
    #prepare the tokens including marks
    for t, v in tokens_marked:
        if len(marked_toks) > 0 and marked_toks[-1][1] in ["+", "-", ">", "<"] and v == "=":
            marked_toks[-1] = (marked_toks[-1][0], marked_toks[-1][1] + v)
        elif len(marked_toks) > 1 and marked_toks[-2][1] == "." and marked_toks[-1][1] in "." and v == ".":
            marked_toks[-2] = (marked_toks[-2][0],"...")
            marked_toks.pop()
        elif len(marked_toks) > 1 and (marked_toks[-2][1], v) in [("\"", "\""), ("\'", "\'")]:
            marked_toks[-2] = (marked_toks[-1][0], marked_toks[-2][1] + marked_toks[-1][1] + v)
            marked_toks.pop()
        else:
            marked_toks.append((t,v))

    #prepare clean and marked tokens
    toks_c_m = []
    for t, v in tokens_cleaned_marked:
        if len(toks_c_m) > 0 and toks_c_m[-1][1] in ["+", "-", ">", "<"] and v == "=":
            toks_c_m[-1] = (toks_c_m[-1][0], toks_c_m[-1][1] + v)
        elif len(toks_c_m) > 1 and toks_c_m[-2][1] == "." and toks_c_m[-1][1] in "." and v == ".":
            toks_c_m[-2] = (toks_c_m[-2][0],"...")
            toks_c_m.pop()
        elif len(toks_c_m) > 1 and (toks_c_m[-2][1], v) in [("\"", "\""), ("\'", "\'")]:
            toks_c_m[-2] = (toks_c_m[-1][0], toks_c_m[-2][1] + toks_c_m[-1][1] + v)
            toks_c_m.pop()
        else:
            toks_c_m.append((t,v))
    
    return marked_toks, model_toks, toks_c_m

def predict_anns(tokens):
    """Predicts annotations for the given tokens and converts them into compilable source code that is written to the output files"""
    marked_toks, model_toks, tokens_cleaned_marked = tokens 
    len_tokens = len(model_toks)
    inputs = np.zeros(len_tokens)
    outputs = np.zeros(len_tokens)

    for i in range(len_tokens):
        inputs[i] = source_dict[model_toks[i]] if model_toks[i] in source_dict else source_dict["_UNKNOWN_"]
    N = len(inputs)
    if N > 4*minibatch_size:
        with open(results_invalid, 'a', encoding="utf-8", newline="") as inv_file:
            writer = csv.writer(inv_file)
            writer.writerow([str(inp), "too many tokens " + str(N)])
        return None
    inputs = scipy.sparse.csr_matrix((np.ones(N, np.float32), (range(N), inputs)), shape=(N, vocab_size))
    outputs = scipy.sparse.csr_matrix((np.ones(N, np.float32), (range(N), outputs)), shape=(N, num_labels))
    sIn = C.io.MinibatchSourceFromData(dict(xx=([inputs], C.layers.typing.Sequence[C.layers.typing.tensor]),
                                            yy=([outputs], C.layers.typing.Sequence[C.layers.typing.tensor])))
    mb = sIn.next_minibatch(N)
    data = {x: mb[sIn.streams['xx']], y: mb[sIn.streams['yy']]}
    
    enhance_data(data, enc)
    pred = dec.eval({x: data[x], t: data[t]})[0]
    count = 0
    func_anns = []
    bracketL_count = bracketR_count = 0
    output_files_comb = []
    skipped_types = set()
    number_of_skipped_types = 0
    number_of_annotated_types = 0
    number_of_none_skips = 0
    number_of_missing_path_skips = 0
    number_of_invalid_types = 0

    #create outputfiles for the combination of original and deeptyper annotations
    for i in range(number_of_predictions):
        output_files_comb.append(open(outp_name + "_combination" + str(i) + ".py", 'w', encoding="utf-8"))

    for ttype, v in marked_toks:
        if v.strip() == "" and v != "\n":
            write_to_files(v, output_files_comb)
            continue
        #check for annotable tokens
        elif v.startswith(("___MF", "___MP")):
            r = [i[0] for i in sorted(enumerate(pred[count]), key=lambda x: x[1], reverse=True)]
            guesses = [target_wl[r[i]][1:-1] for i in range(number_of_predictions)]
            v_value = v[5:-5]
            if v.startswith("___MP"):
                if v_value != "self":
                    write_to_files(v_value, output_files_comb, guesses)
                else:
                    write_to_files(v_value, output_files_comb)
            elif v.startswith("___MF"):
                for guess in guesses:
                    func_anns.append(" -> " + guess)
                write_to_files(v_value, output_files_comb)              
            count += 1
        elif (v.strip() != "" and model_toks[count] == "_UNKNOWN_") or v == model_toks[count] or (ttype in Comment and model_toks[count] == "[comment]") or (ttype in Literal.String.Doc and model_toks[count] == "[docstring]") or (ttype in Literal.Number and model_toks[count] == "[number]") or (ttype in Literal.String and model_toks[count] == "[string]") or (v == "\n" and model_toks[count] == "[EOL]"):
            write_to_files(v, output_files_comb)
            count += 1
        else:
            write_to_files(v, output_files_comb)
        if func_anns:
            if v == '(':
                bracketL_count += 1
            elif v == ')':
                bracketR_count += 1
        #check if function return annotation should be written to file
        if bracketR_count == bracketL_count and bracketL_count > 0:
            if func_anns:
                write_to_files(func_anns, output_files_comb)
            func_anns.clear()
            bracketL_count = bracketR_count = 0

    output_files_deeptyper = []

    #create outputfiles for deeptyper only annotations
    for i in range(number_of_predictions):
        output_files_deeptyper.append(open(outp_name + "_" + str(i) + ".py", 'w', encoding="utf-8"))
    count = 0
    func_anns.clear()
    bracketL_count = bracketR_count = 0
    for ttype, v in tokens_cleaned_marked:
        if v.strip() == "" and v != "\n":
            write_to_files(v, output_files_deeptyper)
            continue
        #check for annotable tokens
        elif v.startswith(("___MF", "___MP")):
            r = [i[0] for i in sorted(enumerate(pred[count]), key=lambda x: x[1], reverse=True)]
            guesses, invalid_guesses, skipped, missing_pathname, none_value, invalid_type = get_predictions(r, skipped_types)
            if v[5:-5] != "self":
                skipped_types.update(invalid_guesses)
                number_of_skipped_types += skipped
                number_of_annotated_types += 1
                number_of_none_skips += none_value
                number_of_missing_path_skips += missing_pathname
                number_of_invalid_types += invalid_type
            v_value = v[5:-5]
            if v.startswith("___MP"):
                if v_value != "self":
                    write_to_files(v_value, output_files_deeptyper, guesses)
                else:
                    write_to_files(v_value, output_files_deeptyper)
            elif v.startswith("___MF"):
                for guess in guesses:
                    func_anns.append(" -> " + guess)
                write_to_files(v_value, output_files_deeptyper)              
            count += 1
        elif (v.strip() != "" and model_toks[count] == "_UNKNOWN_") or v == model_toks[count] or (ttype in Comment and model_toks[count] == "[comment]") or (ttype in Literal.String.Doc and model_toks[count] == "[docstring]") or (ttype in Literal.Number and model_toks[count] == "[number]") or (ttype in Literal.String and model_toks[count] == "[string]") or (v == "\n" and model_toks[count] == "[EOL]"):
            write_to_files(v, output_files_deeptyper)
            count += 1
        else:
            write_to_files(v, output_files_deeptyper)
        if func_anns:
            if v == '(':
                bracketL_count += 1
            elif v == ')':
                bracketR_count += 1
        #check if function annotation should be written to file
        if bracketR_count == bracketL_count and bracketL_count > 0:
            if func_anns:
                write_to_files(func_anns, output_files_deeptyper)
            func_anns.clear()
            bracketL_count = bracketR_count = 0

    print("---------------INVALID TYPES:", skipped_types, "ANNOTATIONS:", number_of_annotated_types*number_of_predictions, "("+str(number_of_annotated_types)+"*"+str(number_of_predictions)+")", "SKIPPED: ",number_of_skipped_types, "---------------")
    #write statistics about invalid and annotated types
    with open(results_invalid, 'a', encoding="utf-8", newline="") as inv_file:
        writer = csv.writer(inv_file)
        writer.writerow([str(inp), str(skipped_types), str(number_of_annotated_types*number_of_predictions), "("+str(number_of_annotated_types)+"*"+str(number_of_predictions)+")", str(number_of_skipped_types), str(number_of_none_skips), str(number_of_invalid_types), str(number_of_missing_path_skips)])

def write_to_files(value, file_list, guesses=None):
    """used to write tokens (and guesses) to multiple files"""
    for i, file in enumerate(file_list):
        if guesses is None:
            if isinstance(value, list):
                file.write(value[i])
            else:
                file.write(value)
        else:
            file.write(value + ": " + guesses[i])

def get_predictions(type_numbers, invalid_types):
    """Returns the top n valid predictions by proofing if they are valid in the specific project environment."""
    valid_guesses = []
    invalid_guesses = []
    missing_pathname = 0
    none_value = 0
    invalid_type = 0
    count = 0
    while len(valid_guesses) < number_of_predictions:
        #get type
        guess = target_wl[type_numbers[count]][1:-1]
        classes_in_guess = set(filter(None, re.split("\[|,", guess)))
        valid = True
        #proof if all types used inside the annotation are valid, important for e.g. lists or Unions
        for klass in classes_in_guess:
            parts = klass.replace("]", "").rsplit(".", 1)
            result = None
            #start subprocess to proof annotation inside project env 
            if len(parts) > 1:
                result = subprocess.run([str(pathlib.Path(__file__).parent.resolve()) + '/is_type_valid.sh', '-m', parts[0], '-c', parts[1]], stdout=subprocess.PIPE)
            else:
                result = subprocess.run([str(pathlib.Path(__file__).parent.resolve()) + '/is_type_valid.sh', '-c', parts[0]], stdout=subprocess.PIPE)
            output = result.stdout.decode('utf-8')
            if "invalid" in output:
                valid = False
                invalid_type += 1
                break
            elif "No module path" in output:
                valid = False
                missing_pathname += 1
                break
            elif "None type" in output:
                valid = False
                none_value += 1
                break
        if valid:
            if guess.startswith("unittest"):
                valid_guesses.append(guess.split(".")[-1])
            else:
                valid_guesses.append(guess)
        else:
            invalid_guesses.append(guess)
        count += 1
    return valid_guesses, invalid_guesses, count - number_of_predictions, missing_pathname, none_value, invalid_type

model = create_model()
enc, dec = model(x, t)
trainer = create_trainer()

with open(inp, 'rb') as f:
    predict_anns(create_tokens(f))