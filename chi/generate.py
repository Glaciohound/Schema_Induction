import argparse
import readline
import os
import atexit
from LM_prompt import get_LM, LM_prompt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--input-file", type=str, default="input.txt")
    parser.add_argument("--output-file", type=str, default="output.txt")
    parser.add_argument("--model-name", type=str, default="bert-large-cased")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return args


def set_readline():
    histfile = os.path.join(os.path.expanduser("~"), ".pyhist")
    if not os.path.exists(histfile):
        open(histfile, 'a').close()
    readline.read_history_file(histfile)
    readline.set_history_length(10000)
    atexit.register(readline.write_history_file, histfile)


def main(args):
    tokenizer, maskedLM = get_LM(args.model_name)

    if args.interactive:
        accumulated = ""
        while True:
            try:
                if accumulated == "":
                    print()
                    print("[[ Input your prompt line ]]")
                    print("----------------------------")
                input_line = input()
                if input_line == "exit":
                    break
                elif input_line == "clear":
                    accumulated = ""
                    continue
                else:
                    if not (accumulated.endswith(' ') or
                            input_line.startswith(' ')):
                        accumulated += ' '
                    predictions, sample_output, status = \
                        LM_prompt(accumulated + input_line,
                                  tokenizer, maskedLM,
                                  args.model_name, args.top_k)
                if status == "success":
                    for one_prediction in predictions:
                        print(one_prediction)
                    if args.verbose:
                        print(sample_output)
                    if accumulated != "":
                        readline.add_history(accumulated + input_line)
                        accumulated = ""
                else:
                    accumulated = sample_output
            except KeyboardInterrupt:
                print("\nTo exit the program, type \"exit\" "
                      "(without the quotation mark).")

    else:
        assert os.path.exists(args.input_file)
        with open(args.input_file, 'r') as f:
            input_lines = f.readlines()
        with open(args.output_file, 'w') as f:
            for input_line in input_lines:
                predictions, sample_output, status = \
                    LM_prompt(input_line, tokenizer, maskedLM, args)
                f.write(input_line+'\n')
                for one_prediction in predictions:
                    f.write(one_prediction+'\n')
                if args.verbose:
                    f.write(sample_output+'\n')


if __name__ == "__main__":
    set_readline()
    args = get_args()
    main(args)
