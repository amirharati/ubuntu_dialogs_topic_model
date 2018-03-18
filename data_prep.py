"""
    write data in one file. Each line one document.
"""
import glob, os
import pathlib

def read_file(file_a):
    """
    `read the files. We try with a heuristic to extract the text.
     In this application it does not matter and we can simply always
     assume the third column is a user id but instead we try to infer it using a simple hack. For this we need to process each file two times.
    """
    lines = [line.strip() for line in open(file_a)]
    user = list()
    possible_user = list()
    text = ""
    for line in lines:
        # first part : date, second: user, third: user or text forth and later text
        parts = line.split()
        # we can add second col for sure but dont know about third col yet.
        user.append(parts[1])
        # some data files dont have any content for some lines
        if len(parts) > 2:
            possible_user.append(parts[2])

    # make unique list
    user = list(set(user))
    possible_user = list((set(possible_user)))
    # check if we have bother users or just one
    if len(user) == 1:
        # we should check the third column.
        # hack: we identify it as a user if it is repeated.
        if len(possible_user) == 1:
            user.append(possible_user[0])
    for line in lines:
        parts = line.split()
        if len(parts) > 2:
            if parts[2] in user:
                if len(parts) > 3:
                    text += " ".join(parts[3:])
            else:
                text += " ".join(parts[2:])
            text += " "
    return(text)


def process_data(inp_dir_a, outfile_a):
    fo = open(outfile_a, "w")
    os.chdir(inp_dir_a)
    for file in glob.glob("*.tsv"):
        print("processing: " + file)
        cont = read_file(file)
        fo.write(cont + "\n")
    fo.close()
    pass


if __name__ == "__main__":
    pathlib.Path('./data').mkdir(exist_ok=True)
    process_data("data/dialogs/4/", "data/dialogs_4.txt")
