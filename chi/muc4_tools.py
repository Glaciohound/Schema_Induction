def print_contents(contents, cased=False):
    for line in contents:
        if cased:
            line = line.lower()
        print(line)
