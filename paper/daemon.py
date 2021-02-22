'''
    This file is for automating the latex compiling procedure.
'''
import os
from time import sleep


def run_shell_interface(shell_string):
    with os.popen(shell_string) as pipe:
        shell_outcome = pipe.read()
    return shell_outcome


def main():
    GET_TEX_FILE_STATUS = 'stat *.tex'
    RFRESH_PDF = 'xelatex *.tex'
    old_description = run_shell_interface(GET_TEX_FILE_STATUS)
    while (True):
        sleep(0.5)
        new_description = run_shell_interface(GET_TEX_FILE_STATUS)
        if (new_description != old_description):
            run_shell_interface(RFRESH_PDF)
            old_description = run_shell_interface(GET_TEX_FILE_STATUS)
            continue
        old_description = new_description


if __name__ == "__main__":
    main()
