import re

def stripper(text):
    return text.strip()


def senna_styler(element):
    argm_re = re.compile("\[ARGM")
    arg_re = re.compile("\[ARG")
    rarg_re = re.compile("\[R-ARG([0-9])")
    carg_re = re.compile("\[C-ARG([0-9])")

    element = re.sub(argm_re, "[AM", element)
    element = re.sub(arg_re, "[A", element)
    element = re.sub(rarg_re, "[R-arg", element)
    element = re.sub(carg_re, "[C-arg", element)

    element = element[1:-1]

    # split at ":"
    element = element.split(":")
    element = list(map(stripper, element))

    return element


def senna_formater(prediction):
    senna_list = []
    for val_dict in prediction:
        text_to_parse = val_dict['description']
        elements = re.findall(re.compile("\[.*?\]"), text_to_parse)
        elements = list(map(senna_styler, elements))
        senna_list.append(dict(elements))

    return senna_list


def get_srl(sentence, srl_predictor):
    prediction = srl_predictor.predict(
        sentence=sentence)
    prediction = prediction['verbs']
    for rel_dict in prediction:
        rel_dict.pop('tags', None)


    return senna_formater(prediction)